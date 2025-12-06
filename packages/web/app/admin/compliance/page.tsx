import { requireAdmin } from '@/lib/auth';
import { GlassCard } from '@/components/ui/GlassCard';
import { GlassButton } from '@/components/ui/GlassButton';
import { Shield, CheckCircle, AlertTriangle, FileText, Download, Clock } from 'lucide-react';

export default async function CompliancePage() {
  await requireAdmin();

  const complianceItems = [
    {
      name: 'SOC 2 Type II',
      status: 'compliant',
      lastAudit: '2024-02-15',
      nextAudit: '2025-02-15',
      coverage: 100
    },
    {
      name: 'GDPR',
      status: 'compliant',
      lastAudit: '2024-01-20',
      nextAudit: '2025-01-20',
      coverage: 100
    },
    {
      name: 'ISO 27001',
      status: 'in_progress',
      lastAudit: '2024-03-01',
      nextAudit: '2024-06-01',
      coverage: 87
    },
    {
      name: 'HIPAA',
      status: 'not_applicable',
      lastAudit: 'N/A',
      nextAudit: 'N/A',
      coverage: 0
    },
    {
      name: 'PCI DSS',
      status: 'compliant',
      lastAudit: '2024-02-28',
      nextAudit: '2025-02-28',
      coverage: 100
    },
  ];

  const auditLogs = [
    { action: 'User data export requested', user: 'admin@latticeforge.ai', timestamp: '2024-03-15 14:32:00', type: 'data_access' },
    { action: 'API key rotated', user: 'security@latticeforge.ai', timestamp: '2024-03-15 12:15:00', type: 'security' },
    { action: 'New admin user added', user: 'admin@latticeforge.ai', timestamp: '2024-03-14 09:45:00', type: 'admin' },
    { action: 'Compliance report generated', user: 'compliance@latticeforge.ai', timestamp: '2024-03-13 16:20:00', type: 'compliance' },
    { action: 'Data retention policy updated', user: 'legal@latticeforge.ai', timestamp: '2024-03-12 11:00:00', type: 'policy' },
  ];

  return (
    <div className="pl-72 p-8">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Compliance</h1>
          <p className="text-slate-400">Security certifications and audit logs</p>
        </div>
        <GlassButton variant="primary" glow>
          <Download className="w-4 h-4 mr-2" />
          Export Report
        </GlassButton>
      </div>

      {/* Compliance Stats */}
      <div className="grid grid-cols-4 gap-6 mb-8">
        <GlassCard blur="heavy" compact>
          <div className="flex items-center gap-3 mb-2">
            <Shield className="w-5 h-5 text-green-400" />
            <span className="text-sm text-slate-400">Certifications</span>
          </div>
          <p className="text-3xl font-bold text-white">{complianceItems.filter(c => c.status === 'compliant').length}</p>
          <p className="text-sm text-green-400 mt-1">Active</p>
        </GlassCard>

        <GlassCard blur="heavy" compact>
          <div className="flex items-center gap-3 mb-2">
            <Clock className="w-5 h-5 text-amber-400" />
            <span className="text-sm text-slate-400">In Progress</span>
          </div>
          <p className="text-3xl font-bold text-white">{complianceItems.filter(c => c.status === 'in_progress').length}</p>
        </GlassCard>

        <GlassCard blur="heavy" compact>
          <div className="flex items-center gap-3 mb-2">
            <FileText className="w-5 h-5 text-blue-400" />
            <span className="text-sm text-slate-400">Audit Logs (30d)</span>
          </div>
          <p className="text-3xl font-bold text-white">1,247</p>
        </GlassCard>

        <GlassCard blur="heavy" compact>
          <div className="flex items-center gap-3 mb-2">
            <AlertTriangle className="w-5 h-5 text-red-400" />
            <span className="text-sm text-slate-400">Open Issues</span>
          </div>
          <p className="text-3xl font-bold text-white">0</p>
        </GlassCard>
      </div>

      <div className="grid grid-cols-2 gap-6 mb-8">
        {/* Compliance Status */}
        <GlassCard blur="heavy">
          <h2 className="text-lg font-bold text-white mb-4">Certification Status</h2>
          <div className="space-y-3">
            {complianceItems.map((item, i) => (
              <div key={i} className="flex items-center justify-between p-4 bg-black/20 rounded-xl border border-white/[0.04]">
                <div className="flex items-center gap-4">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${
                    item.status === 'compliant' ? 'bg-green-500/10' :
                    item.status === 'in_progress' ? 'bg-amber-500/10' :
                    'bg-slate-500/10'
                  }`}>
                    {item.status === 'compliant' ? <CheckCircle className="w-5 h-5 text-green-400" /> :
                     item.status === 'in_progress' ? <Clock className="w-5 h-5 text-amber-400" /> :
                     <Shield className="w-5 h-5 text-slate-400" />}
                  </div>
                  <div>
                    <p className="text-white font-medium">{item.name}</p>
                    <p className="text-sm text-slate-400">
                      {item.status !== 'not_applicable' ? `Next audit: ${item.nextAudit}` : 'Not applicable'}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  {item.status !== 'not_applicable' && (
                    <div className="w-24">
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-slate-400">Coverage</span>
                        <span className="text-white">{item.coverage}%</span>
                      </div>
                      <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${item.coverage === 100 ? 'bg-green-500' : 'bg-amber-500'}`}
                          style={{ width: `${item.coverage}%` }}
                        />
                      </div>
                    </div>
                  )}
                  <span className={`px-2 py-1 rounded text-xs uppercase ${
                    item.status === 'compliant' ? 'bg-green-500/20 text-green-400' :
                    item.status === 'in_progress' ? 'bg-amber-500/20 text-amber-400' :
                    'bg-slate-500/20 text-slate-400'
                  }`}>
                    {item.status.replace('_', ' ')}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </GlassCard>

        {/* Audit Logs */}
        <GlassCard blur="heavy">
          <h2 className="text-lg font-bold text-white mb-4">Recent Audit Logs</h2>
          <div className="space-y-3">
            {auditLogs.map((log, i) => (
              <div key={i} className="p-4 bg-black/20 rounded-xl border border-white/[0.04]">
                <div className="flex items-start justify-between">
                  <div>
                    <p className="text-white font-medium">{log.action}</p>
                    <p className="text-sm text-slate-400">{log.user}</p>
                  </div>
                  <span className={`px-2 py-1 rounded text-xs ${
                    log.type === 'security' ? 'bg-red-500/20 text-red-400' :
                    log.type === 'data_access' ? 'bg-blue-500/20 text-blue-400' :
                    log.type === 'admin' ? 'bg-purple-500/20 text-purple-400' :
                    log.type === 'compliance' ? 'bg-green-500/20 text-green-400' :
                    'bg-amber-500/20 text-amber-400'
                  }`}>
                    {log.type}
                  </span>
                </div>
                <p className="text-xs text-slate-500 mt-2">{log.timestamp}</p>
              </div>
            ))}
          </div>
        </GlassCard>
      </div>
    </div>
  );
}
