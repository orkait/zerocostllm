"use client";
import { useState, useMemo, useCallback } from "react";
import { useRouter } from "next/navigation";
import { Search, Trash2, Download, FileText, Database, MessageSquare } from "lucide-react";
import { ROUTES } from "@/config/routes";
import { PROVIDERS, providerForModel } from "@/config/providers";
import { type ChatThread } from "@/features/chat/lib/chat-db";
import { useThreads, useDeleteThread, useStorageEstimate } from "@/features/chat/api-threads";
import { useChatSessionStore } from "@/stores/chat-session-store";
import { AppShell, ShellStatus } from "@/features/shell/components/app-shell";
import { ShellOverlays } from "@/features/shell/components/shell-overlays";
import { useShellOverlays } from "@/features/shell/hooks/use-shell-overlays";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ProviderAvatar } from "@/components/ui/provider-avatar";
import { cn, formatNumber } from "@/lib/utils";

type StorageEstimate = { usage: number; quota: number };

const EMPTY_COPY = {
  searching: { title: "No matches", hint: "Try a different search" },
  archive: { title: "No saved chats yet", hint: "Open a model and start a conversation" },
} as const;

const MINUTE_MS = 60_000;
const HOUR_MS = 60 * MINUTE_MS;
const DAY_MS = 24 * HOUR_MS;
const MONTH_DAYS = 30;

function formatRelative(ts: number): string {
  const diff = Date.now() - ts;
  const minutes = Math.floor(diff / MINUTE_MS);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(diff / HOUR_MS);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(diff / DAY_MS);
  if (days < MONTH_DAYS) return `${days}d ago`;
  return new Date(ts).toLocaleDateString();
}

function formatBytes(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${n} B`;
}

function downloadFile(filename: string, content: string, type: string) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function exportJson(thread: ChatThread) {
  downloadFile(`${thread.id}.json`, JSON.stringify(thread, null, 2), "application/json");
}

function exportMarkdown(thread: ChatThread) {
  const lines: string[] = [
    `# ${thread.title}`,
    "",
    `- Model: \`${thread.modelId}\``,
    `- Created: ${new Date(thread.createdAt).toISOString()}`,
    `- Updated: ${new Date(thread.updatedAt).toISOString()}`,
    "",
    "---",
    "",
  ];
  for (const message of thread.messages) {
    lines.push(`## ${message.role === "user" ? "User" : "Assistant"}`);
    lines.push("");
    lines.push(message.content);
    lines.push("");
  }
  downloadFile(`${thread.id}.md`, lines.join("\n"), "text/markdown");
}

/** The provider registry owns the mapping: a thread's model id resolves exactly as inference does. */
function providerLabel(modelId: string): string {
  return PROVIDERS[providerForModel(modelId)].label;
}

export default function ChatsPage() {
  const router = useRouter();
  const { openThread } = useChatSessionStore();
  const [query, setQuery] = useState("");
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [bulkConfirm, setBulkConfirm] = useState(false);
  const [bulkError, setBulkError] = useState<string | null>(null);
  const [bulkBusy, setBulkBusy] = useState(false);

  // Server-ish state belongs in React Query, not in a useEffect that setStates on mount. Deleting a
  // thread invalidates the whole thread namespace, so the list and the storage estimate both re-read.
  const { data: threads = [], isLoading: loading } = useThreads(query);
  const { data: estimate = null } = useStorageEstimate();
  const removeThread = useDeleteThread();

  const closeBulkConfirm = useCallback(() => {
    setBulkConfirm(false);
    setBulkError(null);
  }, []);
  const { helpOpen, openHelp, closeHelp } = useShellOverlays({ onEscape: closeBulkConfirm });

  const stats = useMemo(
    () => ({
      total: threads.length,
      totalMessages: threads.reduce((sum, thread) => sum + thread.messages.length, 0),
    }),
    [threads],
  );

  const handleQueryChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setQuery(e.target.value);
  }, []);

  const askBulkDelete = useCallback(() => {
    setBulkError(null);
    setBulkConfirm(true);
  }, []);

  const clearPendingDelete = useCallback(() => setPendingDeleteId(null), []);

  const askDelete = useCallback((thread: ChatThread) => setPendingDeleteId(thread.id), []);

  const confirmDelete = useCallback(
    (thread: ChatThread) => {
      removeThread.mutate(thread.id, { onSuccess: () => setPendingDeleteId(null) });
    },
    [removeThread],
  );

  /** Every thread is attempted, and a failure is reported.
   *
   * This was a sequential `for ... await` with no error handling: the first rejection aborted the
   * loop, left the remaining threads untouched, and left the dialog open showing nothing - the user
   * saw a button that appeared to do nothing. allSettled attempts all of them, and whatever failed
   * is still on screen afterwards for a retry. */
  const confirmBulkDelete = useCallback(() => {
    const removeAll = async () => {
      setBulkBusy(true);
      setBulkError(null);
      const results = await Promise.allSettled(
        threads.map((thread) => removeThread.mutateAsync(thread.id)),
      );
      setBulkBusy(false);
      const failed = results.filter((result) => result.status === "rejected").length;
      if (failed > 0) {
        setBulkError(`${failed} of ${results.length} could not be deleted. The rest were removed.`);
        return;
      }
      setBulkConfirm(false);
    };
    void removeAll();
  }, [threads, removeThread]);

  const resume = useCallback(
    (thread: ChatThread) => {
      openThread(thread.id, thread.modelId);
      router.push(ROUTES.chat);
    },
    [openThread, router],
  );

  return (
    <AppShell
      title="Archive"
      meta={`${stats.total} threads · ${stats.totalMessages} messages`}
      width="prose"
      actions={
        <>
          {estimate ? <StorageBadge estimate={estimate} /> : null}
          <ShellStatus onHelpClick={openHelp} />
        </>
      }
    >
      <div className="flex items-center gap-3 mb-4">
        <div className="relative flex-1">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-(--color-fg-subtle)" />
          <Input
            value={query}
            onChange={handleQueryChange}
            placeholder="Search threads, messages, models..."
            className="pl-8 h-9"
          />
        </div>
        <Button variant="outline" onClick={askBulkDelete} disabled={threads.length === 0}>
          <Trash2 className="w-3.5 h-3.5" /> Delete all
        </Button>
      </div>

      <ThreadList
        threads={threads}
        loading={loading}
        searching={Boolean(query)}
        pendingDeleteId={pendingDeleteId}
        onResume={resume}
        onAskDelete={askDelete}
        onCancelDelete={clearPendingDelete}
        onConfirmDelete={confirmDelete}
      />

      {bulkConfirm ? (
        <BulkDeleteDialog
          count={stats.total}
          busy={bulkBusy}
          error={bulkError}
          onCancel={closeBulkConfirm}
          onConfirm={confirmBulkDelete}
        />
      ) : null}

      <ShellOverlays helpOpen={helpOpen} onCloseHelp={closeHelp} />
    </AppShell>
  );
}

function StorageBadge({ estimate }: { estimate: StorageEstimate }) {
  return (
    <div className="hidden sm:flex items-center gap-1.5 h-7 px-2 rounded-md bg-(--color-surface-1) border border-(--color-border) text-xs font-mono text-(--color-fg-muted)">
      <Database className="w-3 h-3" />
      {formatBytes(estimate.usage)} / {formatBytes(estimate.quota)}
    </div>
  );
}

function ThreadList({
  threads,
  loading,
  searching,
  pendingDeleteId,
  onResume,
  onAskDelete,
  onCancelDelete,
  onConfirmDelete,
}: {
  threads: ChatThread[];
  loading: boolean;
  searching: boolean;
  pendingDeleteId: string | null;
  onResume: (thread: ChatThread) => void;
  onAskDelete: (thread: ChatThread) => void;
  onCancelDelete: () => void;
  onConfirmDelete: (thread: ChatThread) => void;
}) {
  if (loading) {
    return <div className="text-sm text-(--color-fg-subtle) py-8 text-center">Loading...</div>;
  }

  if (threads.length === 0) {
    const copy = searching ? EMPTY_COPY.searching : EMPTY_COPY.archive;
    return (
      <div className="py-16 flex flex-col items-center justify-center gap-3 text-(--color-fg-subtle)">
        <MessageSquare className="w-8 h-8" />
        <div className="text-base font-medium text-(--color-fg-muted)">{copy.title}</div>
        <div className="text-sm">{copy.hint}</div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-(--color-border) bg-(--color-surface-1) divide-y divide-(--color-border)/60">
      {threads.map((thread) => (
        <ThreadRow
          key={thread.id}
          thread={thread}
          pendingDelete={pendingDeleteId === thread.id}
          onResume={onResume}
          onAskDelete={onAskDelete}
          onCancelDelete={onCancelDelete}
          onConfirmDelete={onConfirmDelete}
        />
      ))}
    </div>
  );
}

function ThreadRow({
  thread,
  pendingDelete,
  onResume,
  onAskDelete,
  onCancelDelete,
  onConfirmDelete,
}: {
  thread: ChatThread;
  pendingDelete: boolean;
  onResume: (thread: ChatThread) => void;
  onAskDelete: (thread: ChatThread) => void;
  onCancelDelete: () => void;
  onConfirmDelete: (thread: ChatThread) => void;
}) {
  const handleResume = useCallback(() => onResume(thread), [onResume, thread]);
  const handleAskDelete = useCallback(() => onAskDelete(thread), [onAskDelete, thread]);
  const handleConfirmDelete = useCallback(() => onConfirmDelete(thread), [onConfirmDelete, thread]);
  const handleExportJson = useCallback(() => exportJson(thread), [thread]);
  const handleExportMarkdown = useCallback(() => exportMarkdown(thread), [thread]);

  const tokens = thread.tokenUsage != null ? `${formatNumber(thread.tokenUsage, 0)} tok` : "-";

  return (
    <div
      className={cn(
        "flex items-center gap-3 p-3 hover:bg-(--color-surface-2)/40 transition-colors duration-120",
        pendingDelete && "bg-(--color-danger-soft)",
      )}
    >
      <ProviderAvatar provider={providerLabel(thread.modelId)} size="md" />
      <button type="button" onClick={handleResume} className="flex-1 min-w-0 text-left cursor-pointer">
        <div className="text-sm font-medium truncate">{thread.title}</div>
        <div className="text-xs text-(--color-fg-subtle) flex items-center gap-2 mt-0.5">
          <span className="font-mono truncate">{thread.modelId}</span>
          <span>·</span>
          <span>{thread.messages.length} msg</span>
          <span>·</span>
          <span>{formatRelative(thread.updatedAt)}</span>
        </div>
      </button>
      {pendingDelete ? (
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-(--color-danger)">Delete?</span>
          <Button size="sm" variant="ghost" onClick={onCancelDelete}>
            Cancel
          </Button>
          <Button size="sm" variant="destructive" onClick={handleConfirmDelete}>
            Delete
          </Button>
        </div>
      ) : (
        <div className="flex items-center gap-1">
          <Badge variant="default">{tokens}</Badge>
          <Button variant="ghost" size="icon" aria-label="Export JSON" onClick={handleExportJson}>
            <Download className="w-3.5 h-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            aria-label="Export Markdown"
            onClick={handleExportMarkdown}
          >
            <FileText className="w-3.5 h-3.5" />
          </Button>
          <Button variant="ghost" size="icon" aria-label="Delete" onClick={handleAskDelete}>
            <Trash2 className="w-3.5 h-3.5" />
          </Button>
        </div>
      )}
    </div>
  );
}

function BulkDeleteDialog({
  count,
  busy,
  error,
  onCancel,
  onConfirm,
}: {
  count: number;
  busy: boolean;
  error: string | null;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <div className="fixed inset-0 z-(--z-modal) flex items-center justify-center bg-black/60 backdrop-blur-overlay">
      <div
        role="alertdialog"
        aria-modal="true"
        aria-label="Delete all threads"
        className="rounded-lg bg-(--color-surface-1) border border-(--color-border) p-5 max-w-popover w-full"
      >
        <h2 className="text-lg font-semibold mb-1">Delete all threads?</h2>
        <p className="text-sm text-(--color-fg-muted) mb-4">
          {count} threads will be permanently removed. This cannot be undone.
        </p>
        {error ? <p className="text-sm text-(--color-danger) mb-4">{error}</p> : null}
        <div className="flex gap-2 justify-end">
          <Button variant="outline" onClick={onCancel} disabled={busy}>
            {error ? "Close" : "Cancel"}
          </Button>
          <Button variant="destructive" onClick={onConfirm} disabled={busy}>
            {busy ? "Deleting..." : error ? "Retry" : "Delete all"}
          </Button>
        </div>
      </div>
    </div>
  );
}
