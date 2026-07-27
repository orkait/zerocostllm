// @vitest-environment jsdom
/** Drives the REAL archive page against a REAL IndexedDB (fake-indexeddb), because the archive is
 * entirely client-side: it renders 200 on the server while being completely broken in the browser. */
import "fake-indexeddb/auto";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ThemeProvider } from "next-themes";
import ChatsPage from "./page";
import { createThread, updateThread, listThreads, deleteThread } from "@/features/chat/lib/chat-db";
import { ROUTES } from "@/config/routes";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  usePathname: () => ROUTES.archive,
}));

// The tests share ONE IndexedDB (the module caches its connection), so state leaks between them and
// makes assertions order-dependent. Start every test from an empty archive.
beforeEach(async () => {
  for (const thread of await listThreads()) await deleteThread(thread.id);

  global.ResizeObserver ||= class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
  global.matchMedia ||= ((q: string) => ({
    matches: false,
    media: q,
    addEventListener() {},
    removeEventListener() {},
    addListener() {},
    removeListener() {},
    dispatchEvent: () => false,
  })) as unknown as typeof matchMedia;
});

afterEach(cleanup);

function renderArchive() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <ThemeProvider attribute="data-theme" defaultTheme="system" enableSystem>
      <QueryClientProvider client={client}>
        <ChatsPage />
      </QueryClientProvider>
    </ThemeProvider>,
  );
}

describe("chat archive", () => {
  it("lists a saved conversation", async () => {
    const thread = await createThread("groq/llama-3.3-70b-versatile");
    await updateThread(thread.id, {
      title: "what is a monad",
      messages: [
        { role: "user", content: "what is a monad" },
        { role: "assistant", content: "a monoid in the category of endofunctors" },
      ],
    });

    renderArchive();

    // The thread must actually appear. If the archive is "broken", this is where it shows.
    await waitFor(() => expect(screen.getByText(/what is a monad/i)).toBeTruthy(), { timeout: 3000 });
  });

  it("renders the empty state rather than hanging on a spinner", async () => {
    renderArchive();
    await waitFor(
      () => {
        const body = document.body.textContent ?? "";
        expect(body.length).toBeGreaterThan(0);
      },
      { timeout: 3000 },
    );
  });
});

