/**
 * Study Book GitHub Integration
 *
 * Full GitHub access for the admin Study Book:
 * - Browse repos and files
 * - Read/write code
 * - Create branches, commits, PRs
 * - Review and merge PRs
 * - Manage issues
 *
 * Unlike Copilot which is tied to one repo at a time,
 * we can work across ALL your repos simultaneously.
 */

import { createClient, SupabaseClient } from '@supabase/supabase-js';

// =============================================================================
// TYPES
// =============================================================================

export interface GitHubConnection {
  id?: string;
  user_id: string;
  github_user_id: number;
  github_username: string;
  access_token: string;  // Encrypted in DB
  refresh_token?: string;
  scopes: string[];
  connected_at?: string;
  last_used_at?: string;
}

export interface GitHubRepo {
  id: number;
  name: string;
  full_name: string;
  description: string | null;
  private: boolean;
  html_url: string;
  default_branch: string;
  language: string | null;
  updated_at: string;
  pushed_at: string;
}

export interface GitHubFile {
  name: string;
  path: string;
  sha: string;
  size: number;
  type: 'file' | 'dir';
  content?: string;  // Base64 encoded for files
  download_url?: string;
}

export interface GitHubBranch {
  name: string;
  commit: { sha: string; url: string };
  protected: boolean;
}

export interface GitHubPR {
  id: number;
  number: number;
  title: string;
  body: string;
  state: 'open' | 'closed' | 'merged';
  html_url: string;
  head: { ref: string; sha: string };
  base: { ref: string; sha: string };
  user: { login: string; avatar_url: string };
  created_at: string;
  updated_at: string;
  merged_at?: string;
}

export interface GitHubIssue {
  id: number;
  number: number;
  title: string;
  body: string;
  state: 'open' | 'closed';
  html_url: string;
  user: { login: string; avatar_url: string };
  labels: Array<{ name: string; color: string }>;
  created_at: string;
  updated_at: string;
}

// =============================================================================
// GITHUB CLIENT
// =============================================================================

export class GitHubClient {
  private token: string;
  private baseUrl = 'https://api.github.com';

  constructor(accessToken: string) {
    this.token = accessToken;
  }

  private async fetch<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Authorization': `Bearer ${this.token}`,
        'Accept': 'application/vnd.github.v3+json',
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`GitHub API error ${response.status}: ${error}`);
    }

    return response.json();
  }

  // ---------------------------------------------------------------------------
  // USER
  // ---------------------------------------------------------------------------

  async getUser(): Promise<{ id: number; login: string; name: string; avatar_url: string }> {
    return this.fetch('/user');
  }

  // ---------------------------------------------------------------------------
  // REPOS
  // ---------------------------------------------------------------------------

  async listRepos(options?: {
    type?: 'all' | 'owner' | 'member';
    sort?: 'created' | 'updated' | 'pushed' | 'full_name';
    per_page?: number;
  }): Promise<GitHubRepo[]> {
    const params = new URLSearchParams({
      type: options?.type || 'all',
      sort: options?.sort || 'pushed',
      per_page: String(options?.per_page || 100),
    });
    return this.fetch(`/user/repos?${params}`);
  }

  async getRepo(owner: string, repo: string): Promise<GitHubRepo> {
    return this.fetch(`/repos/${owner}/${repo}`);
  }

  // ---------------------------------------------------------------------------
  // FILES & CONTENTS
  // ---------------------------------------------------------------------------

  async getContents(
    owner: string,
    repo: string,
    path: string = '',
    ref?: string
  ): Promise<GitHubFile | GitHubFile[]> {
    const params = ref ? `?ref=${ref}` : '';
    return this.fetch(`/repos/${owner}/${repo}/contents/${path}${params}`);
  }

  async getFileContent(
    owner: string,
    repo: string,
    path: string,
    ref?: string
  ): Promise<string> {
    const file = await this.getContents(owner, repo, path, ref) as GitHubFile;
    if (file.content) {
      return Buffer.from(file.content, 'base64').toString('utf-8');
    }
    throw new Error('File has no content');
  }

  async createOrUpdateFile(
    owner: string,
    repo: string,
    path: string,
    content: string,
    message: string,
    branch?: string,
    sha?: string  // Required for updates
  ): Promise<{ commit: { sha: string }; content: GitHubFile }> {
    return this.fetch(`/repos/${owner}/${repo}/contents/${path}`, {
      method: 'PUT',
      body: JSON.stringify({
        message,
        content: Buffer.from(content).toString('base64'),
        branch,
        sha,
      }),
    });
  }

  async deleteFile(
    owner: string,
    repo: string,
    path: string,
    message: string,
    sha: string,
    branch?: string
  ): Promise<{ commit: { sha: string } }> {
    return this.fetch(`/repos/${owner}/${repo}/contents/${path}`, {
      method: 'DELETE',
      body: JSON.stringify({
        message,
        sha,
        branch,
      }),
    });
  }

  // ---------------------------------------------------------------------------
  // BRANCHES
  // ---------------------------------------------------------------------------

  async listBranches(owner: string, repo: string): Promise<GitHubBranch[]> {
    return this.fetch(`/repos/${owner}/${repo}/branches`);
  }

  async createBranch(
    owner: string,
    repo: string,
    branchName: string,
    fromSha: string
  ): Promise<{ ref: string; object: { sha: string } }> {
    return this.fetch(`/repos/${owner}/${repo}/git/refs`, {
      method: 'POST',
      body: JSON.stringify({
        ref: `refs/heads/${branchName}`,
        sha: fromSha,
      }),
    });
  }

  async deleteBranch(
    owner: string,
    repo: string,
    branchName: string
  ): Promise<void> {
    await fetch(`${this.baseUrl}/repos/${owner}/${repo}/git/refs/heads/${branchName}`, {
      method: 'DELETE',
      headers: {
        'Authorization': `Bearer ${this.token}`,
      },
    });
  }

  // ---------------------------------------------------------------------------
  // PULL REQUESTS
  // ---------------------------------------------------------------------------

  async listPRs(
    owner: string,
    repo: string,
    state: 'open' | 'closed' | 'all' = 'open'
  ): Promise<GitHubPR[]> {
    return this.fetch(`/repos/${owner}/${repo}/pulls?state=${state}`);
  }

  async getPR(owner: string, repo: string, prNumber: number): Promise<GitHubPR> {
    return this.fetch(`/repos/${owner}/${repo}/pulls/${prNumber}`);
  }

  async createPR(
    owner: string,
    repo: string,
    title: string,
    body: string,
    head: string,
    base: string
  ): Promise<GitHubPR> {
    return this.fetch(`/repos/${owner}/${repo}/pulls`, {
      method: 'POST',
      body: JSON.stringify({ title, body, head, base }),
    });
  }

  async updatePR(
    owner: string,
    repo: string,
    prNumber: number,
    updates: { title?: string; body?: string; state?: 'open' | 'closed' }
  ): Promise<GitHubPR> {
    return this.fetch(`/repos/${owner}/${repo}/pulls/${prNumber}`, {
      method: 'PATCH',
      body: JSON.stringify(updates),
    });
  }

  async mergePR(
    owner: string,
    repo: string,
    prNumber: number,
    commitTitle?: string,
    mergeMethod: 'merge' | 'squash' | 'rebase' = 'squash'
  ): Promise<{ sha: string; merged: boolean }> {
    return this.fetch(`/repos/${owner}/${repo}/pulls/${prNumber}/merge`, {
      method: 'PUT',
      body: JSON.stringify({
        commit_title: commitTitle,
        merge_method: mergeMethod,
      }),
    });
  }

  async getPRDiff(owner: string, repo: string, prNumber: number): Promise<string> {
    const response = await fetch(
      `${this.baseUrl}/repos/${owner}/${repo}/pulls/${prNumber}`,
      {
        headers: {
          'Authorization': `Bearer ${this.token}`,
          'Accept': 'application/vnd.github.v3.diff',
        },
      }
    );
    return response.text();
  }

  async getPRFiles(
    owner: string,
    repo: string,
    prNumber: number
  ): Promise<Array<{
    sha: string;
    filename: string;
    status: string;
    additions: number;
    deletions: number;
    changes: number;
    patch?: string;
  }>> {
    return this.fetch(`/repos/${owner}/${repo}/pulls/${prNumber}/files`);
  }

  // ---------------------------------------------------------------------------
  // ISSUES
  // ---------------------------------------------------------------------------

  async listIssues(
    owner: string,
    repo: string,
    state: 'open' | 'closed' | 'all' = 'open'
  ): Promise<GitHubIssue[]> {
    return this.fetch(`/repos/${owner}/${repo}/issues?state=${state}`);
  }

  async getIssue(owner: string, repo: string, issueNumber: number): Promise<GitHubIssue> {
    return this.fetch(`/repos/${owner}/${repo}/issues/${issueNumber}`);
  }

  async createIssue(
    owner: string,
    repo: string,
    title: string,
    body: string,
    labels?: string[]
  ): Promise<GitHubIssue> {
    return this.fetch(`/repos/${owner}/${repo}/issues`, {
      method: 'POST',
      body: JSON.stringify({ title, body, labels }),
    });
  }

  async updateIssue(
    owner: string,
    repo: string,
    issueNumber: number,
    updates: { title?: string; body?: string; state?: 'open' | 'closed'; labels?: string[] }
  ): Promise<GitHubIssue> {
    return this.fetch(`/repos/${owner}/${repo}/issues/${issueNumber}`, {
      method: 'PATCH',
      body: JSON.stringify(updates),
    });
  }

  async addIssueComment(
    owner: string,
    repo: string,
    issueNumber: number,
    body: string
  ): Promise<{ id: number; body: string }> {
    return this.fetch(`/repos/${owner}/${repo}/issues/${issueNumber}/comments`, {
      method: 'POST',
      body: JSON.stringify({ body }),
    });
  }

  // ---------------------------------------------------------------------------
  // SEARCH
  // ---------------------------------------------------------------------------

  async searchCode(
    query: string,
    options?: { repo?: string; language?: string; per_page?: number }
  ): Promise<{
    total_count: number;
    items: Array<{
      name: string;
      path: string;
      sha: string;
      html_url: string;
      repository: { full_name: string };
    }>;
  }> {
    let q = query;
    if (options?.repo) q += ` repo:${options.repo}`;
    if (options?.language) q += ` language:${options.language}`;

    const params = new URLSearchParams({
      q,
      per_page: String(options?.per_page || 30),
    });

    return this.fetch(`/search/code?${params}`);
  }

  async searchIssues(
    query: string,
    options?: { repo?: string; state?: 'open' | 'closed'; per_page?: number }
  ): Promise<{
    total_count: number;
    items: GitHubIssue[];
  }> {
    let q = query;
    if (options?.repo) q += ` repo:${options.repo}`;
    if (options?.state) q += ` state:${options.state}`;

    const params = new URLSearchParams({
      q,
      per_page: String(options?.per_page || 30),
    });

    return this.fetch(`/search/issues?${params}`);
  }

  // ---------------------------------------------------------------------------
  // COMMITS
  // ---------------------------------------------------------------------------

  async listCommits(
    owner: string,
    repo: string,
    options?: { sha?: string; path?: string; per_page?: number }
  ): Promise<Array<{
    sha: string;
    commit: {
      message: string;
      author: { name: string; date: string };
    };
    html_url: string;
  }>> {
    const params = new URLSearchParams();
    if (options?.sha) params.set('sha', options.sha);
    if (options?.path) params.set('path', options.path);
    if (options?.per_page) params.set('per_page', String(options.per_page));

    const queryString = params.toString();
    return this.fetch(`/repos/${owner}/${repo}/commits${queryString ? `?${queryString}` : ''}`);
  }

  async getCommit(
    owner: string,
    repo: string,
    sha: string
  ): Promise<{
    sha: string;
    commit: { message: string };
    files: Array<{ filename: string; status: string; patch?: string }>;
  }> {
    return this.fetch(`/repos/${owner}/${repo}/commits/${sha}`);
  }

  // ---------------------------------------------------------------------------
  // TREE (for batch operations)
  // ---------------------------------------------------------------------------

  async getTree(
    owner: string,
    repo: string,
    sha: string,
    recursive: boolean = true
  ): Promise<{
    sha: string;
    tree: Array<{
      path: string;
      mode: string;
      type: 'blob' | 'tree';
      sha: string;
      size?: number;
    }>;
  }> {
    return this.fetch(`/repos/${owner}/${repo}/git/trees/${sha}${recursive ? '?recursive=1' : ''}`);
  }
}

// =============================================================================
// CONNECTION MANAGER (Supabase storage)
// =============================================================================

export class GitHubConnectionManager {
  private supabase: SupabaseClient;

  constructor(supabaseUrl?: string, supabaseKey?: string) {
    this.supabase = createClient(
      supabaseUrl || process.env.NEXT_PUBLIC_SUPABASE_URL!,
      supabaseKey || process.env.SUPABASE_SERVICE_ROLE_KEY!
    );
  }

  /**
   * Store GitHub connection after OAuth
   */
  async saveConnection(connection: Omit<GitHubConnection, 'id' | 'connected_at'>): Promise<void> {
    const { error } = await this.supabase
      .from('github_connections')
      .upsert({
        user_id: connection.user_id,
        github_user_id: connection.github_user_id,
        github_username: connection.github_username,
        access_token: connection.access_token,  // Should be encrypted
        refresh_token: connection.refresh_token,
        scopes: connection.scopes,
        connected_at: new Date().toISOString(),
        last_used_at: new Date().toISOString(),
      }, {
        onConflict: 'user_id',
      });

    if (error) throw new Error(`Failed to save GitHub connection: ${error.message}`);
  }

  /**
   * Get GitHub connection for user
   */
  async getConnection(userId: string): Promise<GitHubConnection | null> {
    const { data, error } = await this.supabase
      .from('github_connections')
      .select('*')
      .eq('user_id', userId)
      .single();

    if (error) return null;
    return data;
  }

  /**
   * Get GitHub client for user
   */
  async getClient(userId: string): Promise<GitHubClient | null> {
    const connection = await this.getConnection(userId);
    if (!connection) return null;

    // Update last used
    await this.supabase
      .from('github_connections')
      .update({ last_used_at: new Date().toISOString() })
      .eq('user_id', userId);

    return new GitHubClient(connection.access_token);
  }

  /**
   * Disconnect GitHub
   */
  async disconnect(userId: string): Promise<void> {
    const { error } = await this.supabase
      .from('github_connections')
      .delete()
      .eq('user_id', userId);

    if (error) throw new Error(`Failed to disconnect GitHub: ${error.message}`);
  }
}

// =============================================================================
// SINGLETON
// =============================================================================

let connectionManager: GitHubConnectionManager | null = null;

export function getGitHubConnectionManager(): GitHubConnectionManager {
  if (!connectionManager) {
    connectionManager = new GitHubConnectionManager();
  }
  return connectionManager;
}

// =============================================================================
// DATABASE SCHEMA
// =============================================================================

export const GITHUB_SCHEMA_SQL = `
-- GitHub Connections
CREATE TABLE IF NOT EXISTS github_connections (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID UNIQUE NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  github_user_id BIGINT NOT NULL,
  github_username TEXT NOT NULL,
  access_token TEXT NOT NULL,  -- Consider encryption
  refresh_token TEXT,
  scopes TEXT[] DEFAULT '{}',
  connected_at TIMESTAMPTZ DEFAULT NOW(),
  last_used_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_github_connections_user ON github_connections(user_id);

-- RLS
ALTER TABLE github_connections ENABLE ROW LEVEL SECURITY;

-- Users can only see their own connections
CREATE POLICY github_connections_user_policy ON github_connections
  FOR ALL USING (auth.uid() = user_id);

-- Service role bypass
CREATE POLICY github_connections_service_policy ON github_connections
  FOR ALL USING (auth.jwt()->>'role' = 'service_role');
`;
