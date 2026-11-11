/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Where opt-in usage events are sent. Left unset, usage events are off and never offered. */
  readonly VITE_USAGE_ENDPOINT?: string;
}
