// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import axios from "axios";
import ChatPage from "@/app/page";
import { useChatSessionStore } from "@/stores/chat-session-store";

vi.mock("axios");
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  usePathname: () => "/",
}));

const sendMessage = vi.fn();
vi.mock("@ai-sdk/react", () => ({
  useChat: () => ({
    messages: [],
    sendMessage,
    status: "ready",
    stop: vi.fn(),
    setMessages: vi.fn(),
    error: undefined,
  }),
}));

vi.mock("@/features/chat/lib/chat-db", () => ({
  listThreads: async () => [],
  getThread: async () => undefined,
  createThread: async () => ({}),
  updateThread: async () => undefined,
  deleteThread: async () => undefined,
  searchThreads: async () => [],
  getStorageEstimate: async () => null,
}));

beforeEach(() => {
  global.ResizeObserver ||= class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
  global.matchMedia ||= (() => ({
    matches: false,
    addEventListener() {},
    removeEventListener() {},
    addListener() {},
    removeListener() {},
  })) as unknown as typeof matchMedia;
  vi.mocked(axios.get).mockResolvedValue({ data: [] });
});

afterEach(() => {
  cleanup();
  useChatSessionStore.setState({ chatModelId: null, currentThreadId: null, lastChatModelId: null });
  vi.clearAllMocks();
});

function renderChat() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ChatPage />
    </QueryClientProvider>,
  );
}

describe("chat lives in the app shell", () => {
  it("renders the shell chrome (nav + header), not a second application's", () => {
    useChatSessionStore.setState({ chatModelId: null });
    renderChat();

    // The shell's nav and header, the same ones the market renders.
    expect(screen.getByRole("link", { name: "Market" })).toBeTruthy();
    expect(screen.getByRole("link", { name: "Archive" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Chat", level: 1 })).toBeTruthy();
    expect(screen.getByText(/search models, providers, actions/i)).toBeTruthy();

    // Chat's own panel, in the shell's sidebar.
    expect(screen.getByRole("button", { name: /new chat/i })).toBeTruthy();
    expect(screen.getByPlaceholderText("Search chats...")).toBeTruthy();
  });

  it("with no model picked, offers the market instead of a dead chat", () => {
    useChatSessionStore.setState({ chatModelId: null });
    renderChat();
    expect(screen.getByRole("heading", { name: /pick a model to start/i })).toBeTruthy();
  });

  it("with a model picked, renders the conversation surface", () => {
    useChatSessionStore.setState({ chatModelId: "groq/llama-3.3-70b-versatile" });
    renderChat();

    expect(screen.getByPlaceholderText("Message groq/llama-3.3-70b-versatile...")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Send" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: /how can i help today/i })).toBeTruthy();
    // The old chat header is gone: only the shell's h1 survives.
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
  });
});
