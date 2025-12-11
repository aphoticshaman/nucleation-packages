'use client';

import { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Textarea } from '@/components/ui/textarea';
import {
  Brain,
  Sparkles,
  FlaskConical,
  Code2,
  FileCheck,
  Clock,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  XCircle,
  RefreshCw,
} from 'lucide-react';

// Types
interface InsightReport {
  id: string;
  title: string;
  summary: string;
  target_subject: string;
  current_stage: string;
  status: string;
  confidence_score: number;
  confidence_type: string;
  archaeology_data: Record<string, unknown>;
  nsm_data: Record<string, unknown>;
  theoretical_validation: Record<string, unknown>;
  xyza_data: Record<string, unknown>;
  code_artifacts: Array<{
    filename: string;
    language: string;
    content: string;
    tests_passed: boolean;
  }>;
  impact_analysis: Record<string, unknown>;
  tags: string[];
  created_at: string;
  updated_at: string;
  admin_notes?: string;
  admin_rating?: number;
}

interface InsightStats {
  total_insights: number;
  in_progress: number;
  awaiting_review: number;
  validated: number;
  avg_confidence: number;
}

// Stage config
const STAGE_CONFIG = {
  latent_archaeology: {
    icon: Brain,
    label: 'Archaeology',
    color: 'bg-purple-500',
    description: 'Finding the gradient of ignorance',
  },
  novel_synthesis: {
    icon: Sparkles,
    label: 'Synthesis',
    color: 'bg-blue-500',
    description: 'Creating the fusion',
  },
  theoretical_validation: {
    icon: FlaskConical,
    label: 'Validation',
    color: 'bg-green-500',
    description: 'Proving with math',
  },
  xyza_operationalization: {
    icon: Code2,
    label: 'XYZA',
    color: 'bg-orange-500',
    description: 'Writing the code',
  },
  output_generation: {
    icon: FileCheck,
    label: 'Output',
    color: 'bg-red-500',
    description: 'Final dossier',
  },
};

const STATUS_CONFIG = {
  in_progress: { label: 'In Progress', color: 'bg-yellow-500', icon: Clock },
  awaiting_review: { label: 'Awaiting Review', color: 'bg-blue-500', icon: AlertCircle },
  validated: { label: 'Validated', color: 'bg-green-500', icon: CheckCircle },
  rejected: { label: 'Rejected', color: 'bg-red-500', icon: XCircle },
  needs_revision: { label: 'Needs Revision', color: 'bg-orange-500', icon: RefreshCw },
};

export default function InsightsDashboard() {
  const [insights, setInsights] = useState<InsightReport[]>([]);
  const [stats, setStats] = useState<InsightStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedInsight, setSelectedInsight] = useState<InsightReport | null>(null);
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [filterStage, setFilterStage] = useState<string>('all');

  const fetchInsights = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (filterStatus !== 'all') params.set('status', filterStatus);
      if (filterStage !== 'all') params.set('stage', filterStage);

      const response = await fetch(`/api/elle/insights?${params}`);
      const data = await response.json();

      setInsights(data.insights || []);
      setStats(data.stats);
    } catch (error) {
      console.error('Failed to fetch insights:', error);
    } finally {
      setLoading(false);
    }
  }, [filterStatus, filterStage]);

  useEffect(() => {
    fetchInsights();
  }, [fetchInsights]);

  const handleReview = async (
    insightId: string,
    action: 'validate' | 'reject' | 'revision',
    notes?: string
  ) => {
    const statusMap = {
      validate: 'validated',
      reject: 'rejected',
      revision: 'needs_revision',
    };

    try {
      await fetch('/api/elle/insights', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          insight_id: insightId,
          status: statusMap[action],
          admin_notes: notes,
        }),
      });
      fetchInsights();
      setSelectedInsight(null);
    } catch (error) {
      console.error('Review action failed:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">
          Elle&apos;s Research Insights
        </h1>
        <p className="text-gray-400">
          Autonomous research capture via PROMETHEUS protocol
        </p>
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <Card className="bg-gray-800/50 border-gray-700">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Total Insights</p>
                  <p className="text-2xl font-bold">{stats.total_insights}</p>
                </div>
                <Brain className="h-8 w-8 text-purple-400" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-800/50 border-gray-700">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">In Progress</p>
                  <p className="text-2xl font-bold">{stats.in_progress}</p>
                </div>
                <Clock className="h-8 w-8 text-yellow-400" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-800/50 border-gray-700">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Awaiting Review</p>
                  <p className="text-2xl font-bold">{stats.awaiting_review}</p>
                </div>
                <AlertCircle className="h-8 w-8 text-blue-400" />
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gray-800/50 border-gray-700">
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400 text-sm">Avg Confidence</p>
                  <p className="text-2xl font-bold">
                    {(stats.avg_confidence * 100).toFixed(0)}%
                  </p>
                </div>
                <TrendingUp className="h-8 w-8 text-green-400" />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Pipeline Visualization */}
      <Card className="bg-gray-800/50 border-gray-700 mb-8">
        <CardHeader>
          <CardTitle className="text-lg">PROMETHEUS Pipeline</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            {Object.entries(STAGE_CONFIG).map(([stage, config], index) => {
              const Icon = config.icon;
              const count = insights.filter((i) => i.current_stage === stage).length;

              return (
                <div key={stage} className="flex items-center">
                  <div className="flex flex-col items-center">
                    <div
                      className={`${config.color} p-3 rounded-full mb-2`}
                    >
                      <Icon className="h-6 w-6 text-white" />
                    </div>
                    <span className="text-sm font-medium">{config.label}</span>
                    <span className="text-xs text-gray-400">{count} active</span>
                  </div>
                  {index < Object.keys(STAGE_CONFIG).length - 1 && (
                    <div className="w-16 h-0.5 bg-gray-600 mx-2" />
                  )}
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Filters */}
      <div className="flex gap-4 mb-6">
        <Select value={filterStatus} onValueChange={setFilterStatus}>
          <SelectTrigger className="w-48 bg-gray-800 border-gray-600">
            <SelectValue placeholder="Filter by status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Statuses</SelectItem>
            <SelectItem value="in_progress">In Progress</SelectItem>
            <SelectItem value="awaiting_review">Awaiting Review</SelectItem>
            <SelectItem value="validated">Validated</SelectItem>
            <SelectItem value="rejected">Rejected</SelectItem>
          </SelectContent>
        </Select>

        <Select value={filterStage} onValueChange={setFilterStage}>
          <SelectTrigger className="w-48 bg-gray-800 border-gray-600">
            <SelectValue placeholder="Filter by stage" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Stages</SelectItem>
            {Object.entries(STAGE_CONFIG).map(([stage, config]) => (
              <SelectItem key={stage} value={stage}>
                {config.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Button
          variant="outline"
          onClick={fetchInsights}
          className="border-gray-600"
        >
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Insights List */}
      {loading ? (
        <div className="text-center py-12">
          <div className="animate-spin h-8 w-8 border-2 border-purple-500 border-t-transparent rounded-full mx-auto mb-4" />
          <p className="text-gray-400">Loading insights...</p>
        </div>
      ) : (
        <div className="space-y-4">
          {insights.map((insight) => {
            const stageConfig = STAGE_CONFIG[insight.current_stage as keyof typeof STAGE_CONFIG];
            const statusConfig = STATUS_CONFIG[insight.status as keyof typeof STATUS_CONFIG];
            const StageIcon = stageConfig?.icon || Brain;
            const StatusIcon = statusConfig?.icon || Clock;

            return (
              <Card
                key={insight.id}
                className="bg-gray-800/50 border-gray-700 hover:border-gray-500 transition-colors cursor-pointer"
                onClick={() => setSelectedInsight(insight)}
              >
                <CardContent className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <div className={`${stageConfig?.color || 'bg-gray-500'} p-2 rounded`}>
                          <StageIcon className="h-4 w-4 text-white" />
                        </div>
                        <h3 className="font-semibold text-lg">{insight.title}</h3>
                        <Badge
                          variant="outline"
                          className={`${statusConfig?.color || 'bg-gray-500'} text-white border-0`}
                        >
                          <StatusIcon className="h-3 w-3 mr-1" />
                          {statusConfig?.label || insight.status}
                        </Badge>
                      </div>

                      <p className="text-gray-400 text-sm mb-3">
                        {insight.summary || 'No summary available'}
                      </p>

                      <div className="flex items-center gap-4 text-sm text-gray-500">
                        <span>Target: {insight.target_subject}</span>
                        <span>•</span>
                        <span>
                          Confidence:{' '}
                          {insight.confidence_score
                            ? `${(insight.confidence_score * 100).toFixed(0)}%`
                            : 'N/A'}
                        </span>
                        <span>•</span>
                        <span>
                          {new Date(insight.created_at).toLocaleDateString()}
                        </span>
                      </div>

                      {insight.tags.length > 0 && (
                        <div className="flex gap-2 mt-3">
                          {insight.tags.map((tag) => (
                            <Badge
                              key={tag}
                              variant="secondary"
                              className="bg-gray-700 text-gray-300"
                            >
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      )}
                    </div>

                    {insight.code_artifacts.length > 0 && (
                      <Badge className="bg-green-600">
                        <Code2 className="h-3 w-3 mr-1" />
                        {insight.code_artifacts.length} artifact
                        {insight.code_artifacts.length > 1 ? 's' : ''}
                      </Badge>
                    )}
                  </div>
                </CardContent>
              </Card>
            );
          })}

          {insights.length === 0 && (
            <div className="text-center py-12">
              <Brain className="h-12 w-12 text-gray-600 mx-auto mb-4" />
              <p className="text-gray-400">No insights found</p>
              <p className="text-gray-500 text-sm">
                Elle is working silently. Insights will appear when validated.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Insight Detail Modal */}
      <Dialog
        open={!!selectedInsight}
        onOpenChange={() => setSelectedInsight(null)}
      >
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto bg-gray-800 text-white">
          {selectedInsight && (
            <>
              <DialogHeader>
                <DialogTitle className="text-xl">
                  {selectedInsight.title}
                </DialogTitle>
                <DialogDescription className="text-gray-400">
                  {selectedInsight.summary}
                </DialogDescription>
              </DialogHeader>

              <Tabs defaultValue="overview" className="mt-4">
                <TabsList className="bg-gray-700">
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="archaeology">Archaeology</TabsTrigger>
                  <TabsTrigger value="nsm">NSM</TabsTrigger>
                  <TabsTrigger value="validation">Validation</TabsTrigger>
                  <TabsTrigger value="code">Code</TabsTrigger>
                  <TabsTrigger value="impact">Impact</TabsTrigger>
                </TabsList>

                <TabsContent value="overview" className="mt-4 space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="text-sm text-gray-400">Stage</label>
                      <p className="font-medium">
                        {STAGE_CONFIG[selectedInsight.current_stage as keyof typeof STAGE_CONFIG]?.label}
                      </p>
                    </div>
                    <div>
                      <label className="text-sm text-gray-400">Status</label>
                      <p className="font-medium">{selectedInsight.status}</p>
                    </div>
                    <div>
                      <label className="text-sm text-gray-400">Confidence</label>
                      <p className="font-medium">
                        {selectedInsight.confidence_score
                          ? `${(selectedInsight.confidence_score * 100).toFixed(0)}%`
                          : 'Not assessed'}
                        {selectedInsight.confidence_type && (
                          <span className="text-gray-500 ml-2">
                            ({selectedInsight.confidence_type})
                          </span>
                        )}
                      </p>
                    </div>
                    <div>
                      <label className="text-sm text-gray-400">Target Subject</label>
                      <p className="font-medium">{selectedInsight.target_subject}</p>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="archaeology" className="mt-4">
                  <pre className="bg-gray-900 p-4 rounded overflow-x-auto text-sm">
                    {JSON.stringify(selectedInsight.archaeology_data, null, 2)}
                  </pre>
                </TabsContent>

                <TabsContent value="nsm" className="mt-4">
                  <pre className="bg-gray-900 p-4 rounded overflow-x-auto text-sm">
                    {JSON.stringify(selectedInsight.nsm_data, null, 2)}
                  </pre>
                </TabsContent>

                <TabsContent value="validation" className="mt-4">
                  <pre className="bg-gray-900 p-4 rounded overflow-x-auto text-sm">
                    {JSON.stringify(selectedInsight.theoretical_validation, null, 2)}
                  </pre>
                </TabsContent>

                <TabsContent value="code" className="mt-4 space-y-4">
                  {selectedInsight.code_artifacts.length > 0 ? (
                    selectedInsight.code_artifacts.map((artifact, idx) => (
                      <div key={idx} className="border border-gray-700 rounded">
                        <div className="bg-gray-700 px-4 py-2 flex items-center justify-between">
                          <span className="font-mono text-sm">
                            {artifact.filename}
                          </span>
                          <Badge
                            variant={artifact.tests_passed ? 'default' : 'destructive'}
                          >
                            {artifact.tests_passed ? 'Tests Passed' : 'Tests Failed'}
                          </Badge>
                        </div>
                        <pre className="p-4 overflow-x-auto text-sm">
                          {artifact.content}
                        </pre>
                      </div>
                    ))
                  ) : (
                    <p className="text-gray-400">No code artifacts yet</p>
                  )}
                </TabsContent>

                <TabsContent value="impact" className="mt-4">
                  <pre className="bg-gray-900 p-4 rounded overflow-x-auto text-sm">
                    {JSON.stringify(selectedInsight.impact_analysis, null, 2)}
                  </pre>
                </TabsContent>
              </Tabs>

              {/* Review Actions */}
              {selectedInsight.status === 'awaiting_review' && (
                <div className="mt-6 pt-6 border-t border-gray-700">
                  <h4 className="font-medium mb-4">Admin Review</h4>
                  <Textarea
                    placeholder="Add notes (optional)..."
                    className="bg-gray-900 border-gray-600 mb-4"
                    id="admin-notes"
                  />
                  <div className="flex gap-3">
                    <Button
                      className="bg-green-600 hover:bg-green-700"
                      onClick={() => {
                        const notes = (document.getElementById('admin-notes') as HTMLTextAreaElement)?.value;
                        handleReview(selectedInsight.id, 'validate', notes);
                      }}
                    >
                      <CheckCircle className="h-4 w-4 mr-2" />
                      Validate
                    </Button>
                    <Button
                      variant="outline"
                      className="border-orange-500 text-orange-500"
                      onClick={() => {
                        const notes = (document.getElementById('admin-notes') as HTMLTextAreaElement)?.value;
                        handleReview(selectedInsight.id, 'revision', notes);
                      }}
                    >
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Request Revision
                    </Button>
                    <Button
                      variant="destructive"
                      onClick={() => {
                        const notes = (document.getElementById('admin-notes') as HTMLTextAreaElement)?.value;
                        handleReview(selectedInsight.id, 'reject', notes);
                      }}
                    >
                      <XCircle className="h-4 w-4 mr-2" />
                      Reject
                    </Button>
                  </div>
                </div>
              )}
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
