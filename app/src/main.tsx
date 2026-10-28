import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
// Fonts ship with the app so text looks the same on every OS and offline
import "@fontsource-variable/inter";
import "@fontsource/outfit/700.css";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
