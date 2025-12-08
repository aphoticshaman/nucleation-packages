import { NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { cookies } from 'next/headers';
import { createServerClient } from '@supabase/ssr';

// Service role client
function getServiceClient() {
  return createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!
  );
}

// Verify cron secret for scheduled jobs
function verifyCronAuth(request: Request): boolean {
  const authHeader = request.headers.get('authorization');
  const cronSecret = process.env.CRON_SECRET;
  const isVercelCron = request.headers.get('x-vercel-cron') === '1';

  if (isVercelCron) return true;
  if (cronSecret && authHeader === `Bearer ${cronSecret}`) return true;
  return false;
}

// Server-side auth verification - requires admin role
async function verifyAdminAuth(): Promise<{ isAdmin: boolean; userId?: string; error?: string }> {
  try {
    const cookieStore = await cookies();
    const authClient = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          getAll() {
            return cookieStore.getAll();
          },
          setAll() {
            // Read-only
          },
        },
      }
    );

    const { data: { user }, error: authError } = await authClient.auth.getUser();
    if (authError || !user) {
      return { isAdmin: false, error: 'Authentication required' };
    }

    // Check admin role using service client (bypasses RLS)
    const db = getServiceClient();
    const { data: profile, error: profileError } = await db
      .from('profiles')
      .select('role')
      .eq('id', user.id)
      .single();

    if (profileError || !profile) {
      return { isAdmin: false, userId: user.id, error: 'Profile not found' };
    }

    return { isAdmin: profile.role === 'admin', userId: user.id };
  } catch {
    return { isAdmin: false, error: 'Auth check failed' };
  }
}

interface TrainingExample {
  id: string;
  domain: string;
  input: string;
  output: string;
  quality_score: number;
  weight: number;
  metadata: Record<string, unknown>;
  created_at: string;
}

interface BackupMetadata {
  id: string;
  backup_date: string;
  example_count: number;
  domains: Record<string, number>;
  avg_quality: number;
  file_size_bytes: number;
  checksum: string;
  storage_location: string;
}

// Simple checksum for data integrity
function simpleChecksum(data: string): string {
  let hash = 0;
  for (let i = 0; i < data.length; i++) {
    const char = data.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16).padStart(8, '0');
}

// POST: Create a backup of training data
// SECURITY: Requires admin role or cron secret
export async function POST(request: Request) {
  // Verify authorization - cron jobs or admin users only
  const isCron = verifyCronAuth(request);
  if (!isCron) {
    const auth = await verifyAdminAuth();
    if (!auth.isAdmin) {
      return NextResponse.json(
        { error: auth.error || 'Admin access required' },
        { status: 403 }
      );
    }
  }

  const serviceClient = getServiceClient();
  const today = new Date().toISOString().split('T')[0];

  try {
    // Fetch all training examples
    const { data: examples, error } = await serviceClient
      .from('training_examples')
      .select('*')
      .order('created_at', { ascending: true });

    if (error) throw error;
    if (!examples || examples.length === 0) {
      return NextResponse.json({
        success: false,
        message: 'No training data to backup',
      });
    }

    // Calculate stats
    const domainCounts: Record<string, number> = {};
    let totalQuality = 0;

    (examples as TrainingExample[]).forEach(ex => {
      domainCounts[ex.domain] = (domainCounts[ex.domain] || 0) + 1;
      totalQuality += ex.quality_score;
    });

    // Create backup data in multiple formats
    const alpacaFormat = (examples as TrainingExample[]).map(ex => ({
      instruction: ex.input,
      input: '',
      output: ex.output,
      domain: ex.domain,
      quality: ex.quality_score,
      weight: ex.weight,
      id: ex.id,
      created_at: ex.created_at,
    }));

    const backupJson = JSON.stringify(alpacaFormat, null, 2);
    const checksum = simpleChecksum(backupJson);

    // Store backup in Supabase storage (training-backups bucket)
    const backupFilename = `backup-${today}-${checksum}.json`;

    // Try to upload to storage bucket
    const { error: uploadError } = await serviceClient
      .storage
      .from('training-backups')
      .upload(backupFilename, backupJson, {
        contentType: 'application/json',
        upsert: true,
      });

    let storageLocation = 'supabase:training-backups/' + backupFilename;

    if (uploadError) {
      console.warn('Storage upload failed, saving to training_backups table:', uploadError);
      storageLocation = 'database:training_backups';
    }

    // Also store metadata and compressed backup in a table for redundancy
    const backupMetadata: BackupMetadata = {
      id: `backup-${today}-${checksum}`,
      backup_date: today,
      example_count: examples.length,
      domains: domainCounts,
      avg_quality: totalQuality / examples.length,
      file_size_bytes: new Blob([backupJson]).size,
      checksum,
      storage_location: storageLocation,
    };

    // Upsert backup record
    await serviceClient
      .from('training_backups')
      .upsert({
        id: backupMetadata.id,
        backup_date: backupMetadata.backup_date,
        example_count: backupMetadata.example_count,
        domain_stats: backupMetadata.domains,
        avg_quality: backupMetadata.avg_quality,
        file_size_bytes: backupMetadata.file_size_bytes,
        checksum: backupMetadata.checksum,
        storage_location: backupMetadata.storage_location,
        // Store a sample for quick reference (first 100 examples)
        sample_data: alpacaFormat.slice(0, 100),
        created_at: new Date().toISOString(),
      }, { onConflict: 'id' });

    // Clean up old backups (keep last 30 days)
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

    await serviceClient
      .from('training_backups')
      .delete()
      .lt('backup_date', thirtyDaysAgo.toISOString().split('T')[0]);

    // Also try to clean up storage
    const { data: oldFiles } = await serviceClient
      .storage
      .from('training-backups')
      .list();

    if (oldFiles) {
      const filesToDelete = oldFiles
        .filter(f => {
          const match = f.name.match(/backup-(\d{4}-\d{2}-\d{2})/);
          if (match) {
            return new Date(match[1]) < thirtyDaysAgo;
          }
          return false;
        })
        .map(f => f.name);

      if (filesToDelete.length > 0) {
        await serviceClient
          .storage
          .from('training-backups')
          .remove(filesToDelete);
      }
    }

    return NextResponse.json({
      success: true,
      backup: {
        id: backupMetadata.id,
        date: today,
        exampleCount: examples.length,
        domains: domainCounts,
        avgQuality: (totalQuality / examples.length).toFixed(3),
        fileSizeKb: (backupMetadata.file_size_bytes / 1024).toFixed(2),
        checksum,
        location: storageLocation,
      },
      message: `Backup created with ${examples.length} examples`,
    });
  } catch (error) {
    console.error('Backup error:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Backup failed' },
      { status: 500 }
    );
  }
}

// GET: List available backups
// SECURITY: Requires admin role or cron secret
export async function GET(request: Request) {
  // Verify authorization - cron jobs or admin users only
  const isCron = verifyCronAuth(request);
  if (!isCron) {
    const auth = await verifyAdminAuth();
    if (!auth.isAdmin) {
      return NextResponse.json(
        { error: auth.error || 'Admin access required' },
        { status: 403 }
      );
    }
  }

  const serviceClient = getServiceClient();

  const { data: backups, error } = await serviceClient
    .from('training_backups')
    .select('id, backup_date, example_count, domain_stats, avg_quality, file_size_bytes, checksum, storage_location, created_at')
    .order('backup_date', { ascending: false })
    .limit(30);

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({
    backups,
    totalBackups: backups?.length || 0,
    oldestBackup: backups?.[backups.length - 1]?.backup_date,
    newestBackup: backups?.[0]?.backup_date,
  });
}
