"use client";
import { openDB, type DBSchema, type IDBPDatabase } from "idb";
import { DATABASES } from "@/config/storage";
import { POOL_MODEL_ID } from "@/config/models";

export type ChatMessage = { role: "user" | "assistant"; content: string };

export type ChatThread = {
  id: string;
  modelId: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: ChatMessage[];
  tokenUsage?: number | null;
};

/** The database's name, version and store come from src/config/storage. Renaming any of them points
 *  the app at a fresh, empty database and orphans every existing conversation, so they are declared
 *  once, where that consequence is written down. */
const { name: DB_NAME, version: DB_VERSION, store: STORE } = DATABASES.chat;

const INDEXES = {
  updatedAt: "by-updatedAt",
  modelId: "by-modelId",
} as const;

const TITLE = {
  fallback: "New chat",
  maxLength: 60,
  /** Leaves room for the ellipsis inside maxLength. */
  truncateAt: 57,
} as const;

interface ChatSchema extends DBSchema {
  [STORE]: {
    key: string;
    value: ChatThread;
    indexes: {
      [INDEXES.updatedAt]: number;
      [INDEXES.modelId]: string;
    };
  };
}

let dbPromise: Promise<IDBPDatabase<ChatSchema>> | null = null;

function getDb(): Promise<IDBPDatabase<ChatSchema>> {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("IndexedDB only available in browser"));
  }
  if (!dbPromise) {
    dbPromise = openDB<ChatSchema>(DB_NAME, DB_VERSION, {
      upgrade(db) {
        const store = db.createObjectStore(STORE, { keyPath: "id" });
        store.createIndex(INDEXES.updatedAt, "updatedAt");
        store.createIndex(INDEXES.modelId, "modelId");
      },
    });
  }
  return dbPromise;
}

function newId(): string {
  return `thr_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function deriveTitle(messages: ChatMessage[], fallback: string): string {
  const firstUser = messages.find((message) => message.role === "user");
  if (!firstUser?.content) return fallback;
  const trimmed = firstUser.content.trim().replace(/\s+/g, " ");
  if (trimmed.length <= TITLE.maxLength) return trimmed;
  return `${trimmed.slice(0, TITLE.truncateAt)}...`;
}

export async function createThread(modelId: string): Promise<ChatThread> {
  const now = Date.now();
  const thread: ChatThread = {
    id: newId(),
    modelId,
    title: TITLE.fallback,
    createdAt: now,
    updatedAt: now,
    messages: [],
    tokenUsage: null,
  };
  const db = await getDb();
  await db.put(STORE, thread);
  return thread;
}

export async function getThread(id: string): Promise<ChatThread | undefined> {
  const db = await getDb();
  const raw = await db.get(STORE, id);
  // Same normalization as listThreads: resuming a legacy thread must not crash the chat either.
  return raw ? normalizeThread(raw) : undefined;
}

/** Records written by older builds are not guaranteed to match the current type. TypeScript cannot
 *  enforce anything about bytes that were on disk before this code existed, so every row is
 *  normalized on the way OUT of the database rather than trusted.
 *
 *  This is not hypothetical: a thread missing `modelId` made providerForModel() throw on
 *  `.startsWith`, which unmounted the whole archive and left a blank screen. */
function normalizeThread(raw: ChatThread): ChatThread {
  return {
    ...raw,
    modelId: raw.modelId || POOL_MODEL_ID,
    title: raw.title || TITLE.fallback,
    messages: Array.isArray(raw.messages) ? raw.messages : [],
  };
}

export async function listThreads(): Promise<ChatThread[]> {
  const db = await getDb();
  const all = await db.getAllFromIndex(STORE, INDEXES.updatedAt);
  return all.reverse().map(normalizeThread);
}

export async function updateThread(
  id: string,
  patch: Partial<Omit<ChatThread, "id" | "createdAt">>,
): Promise<ChatThread | undefined> {
  const db = await getDb();
  const existing = await db.get(STORE, id);
  if (!existing) return undefined;
  const next: ChatThread = {
    ...existing,
    ...patch,
    updatedAt: Date.now(),
  };
  if (patch.messages) {
    next.title = patch.messages.length > 0 ? deriveTitle(patch.messages, existing.title) : existing.title;
  }
  await db.put(STORE, next);
  return next;
}

export async function deleteThread(id: string): Promise<void> {
  const db = await getDb();
  await db.delete(STORE, id);
}

export async function searchThreads(query: string): Promise<ChatThread[]> {
  const all = await listThreads();
  const needle = query.trim().toLowerCase();
  if (!needle) return all;
  return all.filter(
    (thread) =>
      thread.title.toLowerCase().includes(needle) ||
      thread.modelId.toLowerCase().includes(needle) ||
      thread.messages.some((message) => message.content.toLowerCase().includes(needle)),
  );
}

export async function getStorageEstimate(): Promise<{ usage: number; quota: number } | null> {
  if (typeof navigator === "undefined" || !navigator.storage?.estimate) return null;
  const estimate = await navigator.storage.estimate();
  return { usage: estimate.usage ?? 0, quota: estimate.quota ?? 0 };
}

