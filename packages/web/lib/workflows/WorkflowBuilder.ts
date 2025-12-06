/**
 * No-Code Workflow Builder
 *
 * Anti-Complaint Spec Section 4.3 Implementation:
 * - Trigger: Alert matches criteria / Schedule / Manual
 * - Enrich: Query external APIs, internal databases, LLM analysis
 * - Decide: If/then logic, confidence thresholds, human approval gates
 * - Act: Create ticket, send notification, update firewall, document finding
 *
 * Visual builder. Version controlled. Shareable across team. Rollback on error.
 */

// =============================================================================
// TYPES
// =============================================================================

export interface Workflow {
  id: string;
  name: string;
  description: string;
  version: number;
  enabled: boolean;

  // Ownership
  createdBy: string;
  createdAt: Date;
  updatedBy: string;
  updatedAt: Date;
  teamId: string;

  // Structure
  trigger: WorkflowTrigger;
  steps: WorkflowStep[];

  // Settings
  settings: WorkflowSettings;

  // History
  lastRunAt?: Date;
  runCount: number;
  successCount: number;
  failureCount: number;
}

export interface WorkflowSettings {
  maxExecutionsPerHour: number;
  timeoutSeconds: number;
  retryOnFailure: boolean;
  maxRetries: number;
  notifyOnFailure: boolean;
  notifyOnSuccess: boolean;
}

// =============================================================================
// TRIGGERS
// =============================================================================

export type WorkflowTrigger =
  | AlertMatchTrigger
  | ScheduleTrigger
  | ManualTrigger
  | WebhookTrigger
  | EventTrigger;

export interface AlertMatchTrigger {
  type: 'alert_match';
  conditions: TriggerCondition[];
  matchMode: 'all' | 'any';
}

export interface ScheduleTrigger {
  type: 'schedule';
  cron: string; // Cron expression
  timezone: string;
}

export interface ManualTrigger {
  type: 'manual';
  allowedUsers: string[] | 'all';
  requireConfirmation: boolean;
}

export interface WebhookTrigger {
  type: 'webhook';
  webhookId: string;
  secret: string;
  validatePayload: boolean;
}

export interface EventTrigger {
  type: 'event';
  eventTypes: string[];
  sourceFilter?: string;
}

export interface TriggerCondition {
  field: string;
  operator: ConditionOperator;
  value: unknown;
}

export type ConditionOperator =
  | 'equals'
  | 'not_equals'
  | 'contains'
  | 'not_contains'
  | 'greater_than'
  | 'less_than'
  | 'in'
  | 'not_in'
  | 'regex'
  | 'exists'
  | 'not_exists';

// =============================================================================
// STEPS
// =============================================================================

export type WorkflowStep =
  | EnrichStep
  | DecideStep
  | ActionStep
  | TransformStep
  | WaitStep;

export interface BaseStep {
  id: string;
  name: string;
  description?: string;
  continueOnError: boolean;
  timeout?: number;
}

// --- Enrich Steps ---

export interface EnrichStep extends BaseStep {
  type: 'enrich';
  enrichmentType: EnrichmentType;
  config: EnrichmentConfig;
  outputVariable: string;
}

export type EnrichmentType =
  | 'api_call'
  | 'database_query'
  | 'llm_analysis'
  | 'threat_lookup'
  | 'whois'
  | 'dns_lookup'
  | 'virustotal'
  | 'shodan'
  | 'internal_intel';

export type EnrichmentConfig =
  | APICallConfig
  | DatabaseQueryConfig
  | LLMAnalysisConfig
  | ThreatLookupConfig;

export interface APICallConfig {
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  url: string;
  headers?: Record<string, string>;
  body?: unknown;
  authentication?: {
    type: 'none' | 'bearer' | 'basic' | 'api_key';
    credentials: string; // Reference to secrets store
  };
}

export interface DatabaseQueryConfig {
  source: 'supabase' | 'internal' | 'custom';
  query: string;
  parameters?: Record<string, unknown>;
}

export interface LLMAnalysisConfig {
  prompt: string;
  model: 'gpt-4' | 'claude-3' | 'internal';
  maxTokens: number;
  temperature: number;
  structuredOutput?: {
    enabled: boolean;
    schema: Record<string, unknown>;
  };
}

export interface ThreatLookupConfig {
  indicator: string; // Variable reference
  sources: ('internal' | 'otx' | 'virustotal' | 'shodan' | 'censys')[];
  includeTTP: boolean;
  includeContext: boolean;
}

// --- Decide Steps ---

export interface DecideStep extends BaseStep {
  type: 'decide';
  decisionType: DecisionType;
  config: DecisionConfig;
  branches: DecisionBranch[];
}

export type DecisionType =
  | 'condition'
  | 'confidence_threshold'
  | 'human_approval'
  | 'routing';

export type DecisionConfig =
  | ConditionDecisionConfig
  | ConfidenceThresholdConfig
  | HumanApprovalConfig
  | RoutingConfig;

export interface ConditionDecisionConfig {
  conditions: TriggerCondition[];
  matchMode: 'all' | 'any';
}

export interface ConfidenceThresholdConfig {
  variable: string;
  thresholds: {
    high: number;
    medium: number;
    low: number;
  };
}

export interface HumanApprovalConfig {
  approvers: string[];
  escalationTimeout: number; // seconds
  escalateTo?: string[];
  requireReason: boolean;
}

export interface RoutingConfig {
  routingKey: string;
  routes: Record<string, string>; // value -> branch name
  defaultBranch: string;
}

export interface DecisionBranch {
  name: string;
  condition?: TriggerCondition[];
  nextSteps: string[]; // Step IDs
}

// --- Action Steps ---

export interface ActionStep extends BaseStep {
  type: 'action';
  actionType: ActionType;
  config: ActionConfig;
}

export type ActionType =
  | 'create_ticket'
  | 'send_notification'
  | 'update_firewall'
  | 'block_indicator'
  | 'document_finding'
  | 'run_playbook'
  | 'webhook_call'
  | 'update_case'
  | 'escalate';

export type ActionConfig =
  | CreateTicketConfig
  | SendNotificationConfig
  | UpdateFirewallConfig
  | BlockIndicatorConfig
  | DocumentFindingConfig
  | WebhookCallConfig;

export interface CreateTicketConfig {
  system: 'jira' | 'servicenow' | 'pagerduty' | 'opsgenie' | 'linear';
  project: string;
  issueType: string;
  title: string;
  description: string;
  priority: string;
  assignee?: string;
  labels?: string[];
  customFields?: Record<string, unknown>;
}

export interface SendNotificationConfig {
  channels: NotificationChannel[];
  title: string;
  body: string;
  priority: 'low' | 'normal' | 'high' | 'urgent';
  attachData?: boolean;
}

export interface NotificationChannel {
  type: 'email' | 'slack' | 'teams' | 'discord' | 'pagerduty' | 'sms';
  target: string; // Email, channel ID, phone number, etc.
}

export interface UpdateFirewallConfig {
  integration: string;
  action: 'block' | 'allow' | 'monitor';
  indicators: string[]; // Variable references
  duration?: number; // seconds, undefined = permanent
  comment?: string;
}

export interface BlockIndicatorConfig {
  indicatorVariable: string;
  blockLists: string[];
  expirationHours?: number;
  reason: string;
}

export interface DocumentFindingConfig {
  caseId?: string;
  title: string;
  content: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  tags: string[];
  attachments?: string[];
}

export interface WebhookCallConfig {
  url: string;
  method: 'POST' | 'PUT';
  headers?: Record<string, string>;
  payload: unknown;
  retryOnFailure: boolean;
}

// --- Transform Steps ---

export interface TransformStep extends BaseStep {
  type: 'transform';
  transformType: 'map' | 'filter' | 'aggregate' | 'format';
  inputVariable: string;
  outputVariable: string;
  config: unknown;
}

// --- Wait Steps ---

export interface WaitStep extends BaseStep {
  type: 'wait';
  waitType: 'delay' | 'until' | 'for_event';
  config: WaitConfig;
}

export type WaitConfig =
  | { type: 'delay'; seconds: number }
  | { type: 'until'; condition: TriggerCondition[] }
  | { type: 'for_event'; eventType: string; timeout: number };

// =============================================================================
// EXECUTION
// =============================================================================

export interface WorkflowExecution {
  id: string;
  workflowId: string;
  workflowVersion: number;
  status: ExecutionStatus;

  // Trigger info
  triggeredAt: Date;
  triggeredBy: string;
  triggerData: unknown;

  // Execution state
  currentStep?: string;
  stepResults: StepResult[];
  variables: Record<string, unknown>;

  // Completion
  completedAt?: Date;
  duration?: number; // milliseconds
  error?: string;
}

export type ExecutionStatus =
  | 'pending'
  | 'running'
  | 'waiting_approval'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'timed_out';

export interface StepResult {
  stepId: string;
  stepName: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  startedAt?: Date;
  completedAt?: Date;
  duration?: number;
  output?: unknown;
  error?: string;
}

// =============================================================================
// WORKFLOW ENGINE
// =============================================================================

export class WorkflowEngine {
  private workflows: Map<string, Workflow> = new Map();
  private executions: Map<string, WorkflowExecution> = new Map();

  /**
   * Register a workflow.
   */
  registerWorkflow(workflow: Workflow): void {
    this.workflows.set(workflow.id, workflow);
  }

  /**
   * Execute a workflow.
   */
  async execute(
    workflowId: string,
    triggerData: unknown
  ): Promise<WorkflowExecution> {
    const workflow = this.workflows.get(workflowId);
    if (!workflow) {
      throw new Error(`Workflow ${workflowId} not found`);
    }

    if (!workflow.enabled) {
      throw new Error(`Workflow ${workflowId} is disabled`);
    }

    // Create execution record
    const execution: WorkflowExecution = {
      id: `exec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      workflowId,
      workflowVersion: workflow.version,
      status: 'running',
      triggeredAt: new Date(),
      triggeredBy: 'system',
      triggerData,
      stepResults: [],
      variables: { trigger: triggerData },
    };

    this.executions.set(execution.id, execution);

    try {
      // Execute steps
      for (const step of workflow.steps) {
        execution.currentStep = step.id;

        const stepResult = await this.executeStep(step, execution);
        execution.stepResults.push(stepResult);

        if (stepResult.status === 'failed' && !step.continueOnError) {
          throw new Error(stepResult.error || `Step ${step.name} failed`);
        }
      }

      execution.status = 'completed';
      execution.completedAt = new Date();
      execution.duration = execution.completedAt.getTime() - execution.triggeredAt.getTime();

      // Update workflow stats
      workflow.lastRunAt = new Date();
      workflow.runCount++;
      workflow.successCount++;
    } catch (error) {
      execution.status = 'failed';
      execution.error = error instanceof Error ? error.message : 'Unknown error';
      execution.completedAt = new Date();
      execution.duration = execution.completedAt.getTime() - execution.triggeredAt.getTime();

      workflow.runCount++;
      workflow.failureCount++;
    }

    return execution;
  }

  /**
   * Execute a single step.
   */
  private async executeStep(
    step: WorkflowStep,
    execution: WorkflowExecution
  ): Promise<StepResult> {
    const result: StepResult = {
      stepId: step.id,
      stepName: step.name,
      status: 'running',
      startedAt: new Date(),
    };

    try {
      switch (step.type) {
        case 'enrich':
          result.output = await this.executeEnrichStep(step, execution);
          break;
        case 'decide':
          result.output = await this.executeDecideStep(step, execution);
          break;
        case 'action':
          result.output = await this.executeActionStep(step, execution);
          break;
        case 'transform':
          result.output = await this.executeTransformStep(step, execution);
          break;
        case 'wait':
          result.output = await this.executeWaitStep(step, execution);
          break;
      }

      result.status = 'completed';
    } catch (error) {
      result.status = 'failed';
      result.error = error instanceof Error ? error.message : 'Unknown error';
    }

    result.completedAt = new Date();
    result.duration = result.completedAt.getTime() - (result.startedAt?.getTime() || 0);

    return result;
  }

  private async executeEnrichStep(
    step: EnrichStep,
    execution: WorkflowExecution
  ): Promise<unknown> {
    // In production, this would actually call APIs, databases, etc.
    // For now, return mock enrichment data
    const enrichment = {
      type: step.enrichmentType,
      timestamp: new Date().toISOString(),
      data: { mock: true },
    };

    // Store in execution variables
    execution.variables[step.outputVariable] = enrichment;

    return enrichment;
  }

  private async executeDecideStep(
    step: DecideStep,
    execution: WorkflowExecution
  ): Promise<{ branch: string }> {
    // Evaluate conditions and return selected branch
    // For now, return first branch
    return { branch: step.branches[0]?.name || 'default' };
  }

  private async executeActionStep(
    step: ActionStep,
    execution: WorkflowExecution
  ): Promise<unknown> {
    // In production, this would create tickets, send notifications, etc.
    return {
      action: step.actionType,
      timestamp: new Date().toISOString(),
      success: true,
    };
  }

  private async executeTransformStep(
    step: TransformStep,
    execution: WorkflowExecution
  ): Promise<unknown> {
    const input = execution.variables[step.inputVariable];
    // Apply transformation
    execution.variables[step.outputVariable] = input;
    return input;
  }

  private async executeWaitStep(
    step: WaitStep,
    execution: WorkflowExecution
  ): Promise<void> {
    if (step.config.type === 'delay') {
      await new Promise((resolve) => setTimeout(resolve, step.config.seconds * 1000));
    }
    // Other wait types would be handled differently
  }

  /**
   * Get execution status.
   */
  getExecution(executionId: string): WorkflowExecution | undefined {
    return this.executions.get(executionId);
  }

  /**
   * Cancel a running execution.
   */
  cancelExecution(executionId: string): boolean {
    const execution = this.executions.get(executionId);
    if (!execution || execution.status !== 'running') {
      return false;
    }

    execution.status = 'cancelled';
    execution.completedAt = new Date();
    return true;
  }
}

// =============================================================================
// FACTORY
// =============================================================================

export function createWorkflow(
  name: string,
  description: string,
  trigger: WorkflowTrigger,
  createdBy: string,
  teamId: string
): Workflow {
  return {
    id: `wf_${Date.now()}`,
    name,
    description,
    version: 1,
    enabled: false,
    createdBy,
    createdAt: new Date(),
    updatedBy: createdBy,
    updatedAt: new Date(),
    teamId,
    trigger,
    steps: [],
    settings: {
      maxExecutionsPerHour: 100,
      timeoutSeconds: 300,
      retryOnFailure: true,
      maxRetries: 3,
      notifyOnFailure: true,
      notifyOnSuccess: false,
    },
    runCount: 0,
    successCount: 0,
    failureCount: 0,
  };
}

export const workflowEngine = new WorkflowEngine();
