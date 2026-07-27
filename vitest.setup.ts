// The app shell mounts VaultIndicator on every surface, and that calls into IndexedDB on mount.
// jsdom has no IndexedDB, so any test that renders a shell produced an unhandled
// "ReferenceError: indexedDB is not defined" rejection - the suite reported 114 passing tests and
// still exited 1. Registering the polyfill globally fixes it for every test rather than asking each
// one to remember to mock a store it never mentions.
import "fake-indexeddb/auto";
