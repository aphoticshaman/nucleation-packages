'use client';

import { useState, useMemo } from 'react';

type SortDirection = 'asc' | 'desc' | null;

interface Column<T> {
  key: keyof T | string;
  label: string;
  sortable?: boolean;
  width?: string;
  render?: (value: unknown, row: T, index: number) => React.ReactNode;
  align?: 'left' | 'center' | 'right';
}

interface DataTableProps<T extends Record<string, unknown>> {
  data: T[];
  columns: Column<T>[];
  onRowClick?: (row: T, index: number) => void;
  selectable?: boolean;
  selectedRows?: Set<number>;
  onSelectionChange?: (selected: Set<number>) => void;
  stickyHeader?: boolean;
  maxHeight?: string;
  emptyMessage?: string;
  loading?: boolean;
}

// Component 15: Sortable Data Table with Risk Coloring
export function DataTable<T extends Record<string, unknown>>({
  data,
  columns,
  onRowClick,
  selectable = false,
  selectedRows = new Set(),
  onSelectionChange,
  stickyHeader = true,
  maxHeight = '400px',
  emptyMessage = 'No data available',
  loading = false,
}: DataTableProps<T>) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<SortDirection>(null);

  // Handle sort
  const handleSort = (key: string) => {
    if (sortKey === key) {
      if (sortDirection === 'asc') {
        setSortDirection('desc');
      } else if (sortDirection === 'desc') {
        setSortKey(null);
        setSortDirection(null);
      }
    } else {
      setSortKey(key);
      setSortDirection('asc');
    }
  };

  // Sort data
  const sortedData = useMemo(() => {
    if (!sortKey || !sortDirection) return data;

    return [...data].sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];

      if (aVal === bVal) return 0;
      if (aVal === null || aVal === undefined) return 1;
      if (bVal === null || bVal === undefined) return -1;

      let comparison = 0;
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        comparison = aVal - bVal;
      } else {
        comparison = String(aVal).localeCompare(String(bVal));
      }

      return sortDirection === 'asc' ? comparison : -comparison;
    });
  }, [data, sortKey, sortDirection]);

  // Handle selection
  const toggleRow = (index: number) => {
    const newSelected = new Set(selectedRows);
    if (newSelected.has(index)) {
      newSelected.delete(index);
    } else {
      newSelected.add(index);
    }
    onSelectionChange?.(newSelected);
  };

  const toggleAll = () => {
    if (selectedRows.size === data.length) {
      onSelectionChange?.(new Set());
    } else {
      onSelectionChange?.(new Set(data.map((_, i) => i)));
    }
  };

  return (
    <div
      className="relative overflow-auto rounded-lg border border-slate-700 bg-slate-900/50"
      style={{ maxHeight }}
    >
      {loading && (
        <div className="absolute inset-0 bg-slate-900/80 flex items-center justify-center z-20">
          <div className="flex items-center gap-2 text-cyan-400">
            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
                fill="none"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
              />
            </svg>
            <span className="text-sm">Loading...</span>
          </div>
        </div>
      )}

      <table className="w-full text-sm">
        <thead className={stickyHeader ? 'sticky top-0 z-10' : ''}>
          <tr className="bg-slate-800 border-b border-slate-700">
            {selectable && (
              <th className="px-3 py-2 w-10">
                <input
                  type="checkbox"
                  checked={selectedRows.size === data.length && data.length > 0}
                  onChange={toggleAll}
                  className="rounded border-slate-600 bg-slate-700 text-cyan-500 focus:ring-cyan-500"
                />
              </th>
            )}
            {columns.map((col) => (
              <th
                key={String(col.key)}
                className={`
                  px-3 py-2 text-slate-300 font-medium
                  ${col.align === 'center' ? 'text-center' : col.align === 'right' ? 'text-right' : 'text-left'}
                  ${col.sortable ? 'cursor-pointer hover:text-cyan-400 select-none' : ''}
                `}
                style={{ width: col.width }}
                onClick={() => col.sortable && handleSort(String(col.key))}
              >
                <div className="flex items-center gap-1">
                  <span>{col.label}</span>
                  {col.sortable && (
                    <span className="text-slate-500">
                      {sortKey === col.key ? (
                        sortDirection === 'asc' ? '↑' : '↓'
                      ) : '↕'}
                    </span>
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedData.length === 0 ? (
            <tr>
              <td
                colSpan={columns.length + (selectable ? 1 : 0)}
                className="px-3 py-8 text-center text-slate-500"
              >
                {emptyMessage}
              </td>
            </tr>
          ) : (
            sortedData.map((row, rowIndex) => (
              <tr
                key={rowIndex}
                onClick={() => onRowClick?.(row, rowIndex)}
                className={`
                  border-b border-slate-800 transition-colors
                  ${onRowClick ? 'cursor-pointer hover:bg-slate-800/50' : ''}
                  ${selectedRows.has(rowIndex) ? 'bg-cyan-900/20' : ''}
                `}
              >
                {selectable && (
                  <td className="px-3 py-2">
                    <input
                      type="checkbox"
                      checked={selectedRows.has(rowIndex)}
                      onChange={() => toggleRow(rowIndex)}
                      onClick={(e) => e.stopPropagation()}
                      className="rounded border-slate-600 bg-slate-700 text-cyan-500 focus:ring-cyan-500"
                    />
                  </td>
                )}
                {columns.map((col) => {
                  const value = row[col.key as keyof T];
                  return (
                    <td
                      key={String(col.key)}
                      className={`
                        px-3 py-2 text-slate-300
                        ${col.align === 'center' ? 'text-center' : col.align === 'right' ? 'text-right' : 'text-left'}
                      `}
                    >
                      {col.render ? col.render(value, row, rowIndex) : String(value ?? '')}
                    </td>
                  );
                })}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

// Risk-colored cell renderer
export function RiskCell({ value }: { value: number }) {
  const color = value >= 0.8 ? 'text-red-400' :
                value >= 0.6 ? 'text-orange-400' :
                value >= 0.4 ? 'text-yellow-400' :
                'text-green-400';

  const bg = value >= 0.8 ? 'bg-red-500/10' :
             value >= 0.6 ? 'bg-orange-500/10' :
             value >= 0.4 ? 'bg-yellow-500/10' :
             'bg-green-500/10';

  return (
    <span className={`px-2 py-0.5 rounded font-mono ${color} ${bg}`}>
      {(value * 100).toFixed(0)}%
    </span>
  );
}

// Trend cell renderer
export function TrendCell({ value, previousValue }: { value: number; previousValue: number }) {
  const delta = value - previousValue;
  const isUp = delta > 0;
  const isFlat = Math.abs(delta) < 0.01;

  return (
    <div className="flex items-center gap-1">
      <span className="font-mono">{value.toFixed(2)}</span>
      {!isFlat && (
        <span className={isUp ? 'text-red-400' : 'text-green-400'}>
          {isUp ? '↑' : '↓'}
          {Math.abs(delta * 100).toFixed(0)}%
        </span>
      )}
    </div>
  );
}

// Status badge cell
export function StatusCell({ status }: { status: 'active' | 'monitoring' | 'resolved' | 'escalated' }) {
  const config = {
    active: { bg: 'bg-yellow-500/20', text: 'text-yellow-400', label: 'ACTIVE' },
    monitoring: { bg: 'bg-cyan-500/20', text: 'text-cyan-400', label: 'MONITORING' },
    resolved: { bg: 'bg-green-500/20', text: 'text-green-400', label: 'RESOLVED' },
    escalated: { bg: 'bg-red-500/20', text: 'text-red-400', label: 'ESCALATED' },
  };

  const { bg, text, label } = config[status];

  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${bg} ${text}`}>
      {label}
    </span>
  );
}

// Example data type
export interface SignalRow {
  id: string;
  timestamp: string;
  headline: string;
  source: string;
  domain: string;
  riskScore: number;
  confidence: number;
  status: 'active' | 'monitoring' | 'resolved' | 'escalated';
}

// Example columns configuration
export const signalColumns: Column<SignalRow>[] = [
  { key: 'timestamp', label: 'Time', sortable: true, width: '120px' },
  { key: 'headline', label: 'Headline', sortable: true },
  { key: 'source', label: 'Source', sortable: true, width: '100px' },
  { key: 'domain', label: 'Domain', sortable: true, width: '100px' },
  {
    key: 'riskScore',
    label: 'Risk',
    sortable: true,
    width: '80px',
    align: 'center',
    render: (val) => <RiskCell value={val as number} />,
  },
  {
    key: 'confidence',
    label: 'Conf.',
    sortable: true,
    width: '70px',
    align: 'center',
    render: (val) => `${((val as number) * 100).toFixed(0)}%`,
  },
  {
    key: 'status',
    label: 'Status',
    sortable: true,
    width: '100px',
    render: (val) => <StatusCell status={val as SignalRow['status']} />,
  },
];
