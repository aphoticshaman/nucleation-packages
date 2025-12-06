import { requireEnterprise, createClient } from '@/lib/auth';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';
import { Users, UserPlus, Mail, Shield } from 'lucide-react';

export default async function TeamPage() {
  const user = await requireEnterprise();
  const supabase = await createClient();

  const { data: org } = await supabase
    .from('organizations')
    .select('*')
    .eq('id', user.organization_id)
    .single();

  const { data: teamMembers } = await supabase
    .from('profiles')
    .select('*')
    .eq('organization_id', user.organization_id);

  return (
    <div>
      <div className="mb-6 md:mb-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl md:text-2xl font-bold text-white">Team</h1>
          <p className="text-slate-400 text-sm md:text-base">
            Manage team members and permissions
          </p>
        </div>
        <GlassButton variant="primary" glow>
          <UserPlus className="w-4 h-4 mr-2" />
          Invite Member
        </GlassButton>
      </div>

      {/* Team Stats */}
      <div className="grid grid-cols-2 gap-4 md:gap-6 mb-6 md:mb-8">
        <GlassCard blur="heavy" compact>
          <div className="flex items-center gap-3 mb-3">
            <Users className="w-5 h-5 text-blue-400" />
            <span className="text-sm text-slate-400">Team Members</span>
          </div>
          <p className="text-2xl md:text-3xl font-bold text-white">
            {teamMembers?.length || 0}
          </p>
          <p className="text-sm text-slate-500 mt-1">
            of {org?.team_seats_limit || 5} seats
          </p>
        </GlassCard>

        <GlassCard blur="heavy" compact>
          <div className="flex items-center gap-3 mb-3">
            <Mail className="w-5 h-5 text-amber-400" />
            <span className="text-sm text-slate-400">Pending Invites</span>
          </div>
          <p className="text-2xl md:text-3xl font-bold text-white">0</p>
          <p className="text-sm text-slate-500 mt-1">awaiting response</p>
        </GlassCard>
      </div>

      {/* Team Members List */}
      <GlassCard blur="heavy">
        <h2 className="text-lg font-bold text-white mb-4">Members</h2>
        <div className="space-y-3">
          {teamMembers?.map((member) => (
            <div
              key={member.id}
              className="flex items-center justify-between p-4 bg-black/20 rounded-xl border border-white/[0.04]"
            >
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 rounded-full bg-emerald-600 flex items-center justify-center text-white font-bold">
                  {member.full_name?.[0] || member.email[0].toUpperCase()}
                </div>
                <div>
                  <p className="text-white font-medium">{member.full_name || 'Unnamed'}</p>
                  <p className="text-sm text-slate-400">{member.email}</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <span className="flex items-center gap-1 text-sm text-slate-400">
                  <Shield className="w-4 h-4" />
                  {member.role || 'Member'}
                </span>
              </div>
            </div>
          )) || (
            <p className="text-slate-400 text-center py-8">No team members found</p>
          )}
        </div>
      </GlassCard>
    </div>
  );
}
