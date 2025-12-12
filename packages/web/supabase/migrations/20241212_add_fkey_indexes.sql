-- Add missing foreign key indexes for performance
-- These indexes improve JOIN and DELETE performance on foreign key columns

-- insight_reports
CREATE INDEX IF NOT EXISTS idx_insight_reports_parent_insight
    ON insight_reports(parent_insight);

CREATE INDEX IF NOT EXISTS idx_insight_reports_reviewed_by
    ON insight_reports(reviewed_by);

-- training_export_batches
CREATE INDEX IF NOT EXISTS idx_training_export_batches_created_by
    ON training_export_batches(created_by);

-- training_items
CREATE INDEX IF NOT EXISTS idx_training_items_created_by
    ON training_items(created_by);

CREATE INDEX IF NOT EXISTS idx_training_items_deleted_by
    ON training_items(deleted_by);

CREATE INDEX IF NOT EXISTS idx_training_items_exported_by
    ON training_items(exported_by);

CREATE INDEX IF NOT EXISTS idx_training_items_reviewed_by
    ON training_items(reviewed_by);
