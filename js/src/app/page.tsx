"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import styles from "./page.module.css";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") ||
  "http://localhost:8000";

const DOMAINS = ["research", "blog", "documentation"] as const;
type Domain = (typeof DOMAINS)[number];

const MODES = [
  { id: "search", label: "Search" },
  { id: "deep", label: "Deep Research (RAG)" },
  { id: "similar", label: "Find Similar" },
] as const;
type Mode = (typeof MODES)[number]["id"];

const SIDEBAR_MODES = MODES.filter((mode) => mode.id !== "similar");

type SearchResult = {
  id?: string | number;
  doc_id?: string | number;
  index?: string | number;
  title?: string;
  domain?: string;
  source?: string;
  score?: number;
  snippet?: string;
  text?: string;
};

type SearchResponse = {
  results?: SearchResult[];
  total?: number;
  answer?: string;
};

type DocDetails = {
  id?: string | number;
  title?: string;
  domain?: string;
  source?: string;
  content?: string;
  text?: string;
};

type UrlState = {
  domain: Domain;
  mode: Mode;
  query: string;
  seedId: string;
  offset: number;
  limit: number;
  docId: string;
  sidebarCollapsed: boolean;
  mock: boolean;
};

function parseDomain(value: string | null): Domain {
  if (value && DOMAINS.includes(value as Domain)) {
    return value as Domain;
  }
  return "research";
}

function parseMode(value: string | null): Mode {
  if (value && MODES.some((mode) => mode.id === value)) {
    return value as Mode;
  }
  return "search";
}

function parseNonNegativeInt(value: string | null, fallback: number): number {
  const next = Number(value);
  if (!Number.isFinite(next)) {
    return fallback;
  }
  return next >= 0 ? Math.floor(next) : fallback;
}

function parsePositiveInt(value: string | null, fallback: number): number {
  const next = Number(value);
  if (!Number.isFinite(next)) {
    return fallback;
  }
  return next > 0 ? Math.floor(next) : fallback;
}

function getDocId(result: SearchResult): string | number | undefined {
  return result.id ?? result.doc_id ?? result.index;
}

function getPreview(result: SearchResult): string {
  const text = result.snippet || result.text || "";
  if (!text) {
    return "";
  }
  return text.length > 240 ? `${text.slice(0, 240)}...` : text;
}

function createFakeResults(domain: Domain, limit: number): SearchResult[] {
  const topics = [
    "vector compression",
    "drift detection",
    "retrieval pipelines",
    "multimodal scoring",
    "embedding health",
    "calibration checks",
    "segment parity",
    "prompt routing",
    "memory indexing",
    "citation ranking",
  ];

  return Array.from({ length: limit }, (_, index) => {
    const topic = topics[Math.floor(Math.random() * topics.length)];
    const score = 0.72 + Math.random() * 0.27;
    return {
      id: `mock-${domain}-${index + 1}`,
      title: `${domain} insights: ${topic}`,
      domain,
      source: "SemantikML Lab",
      score,
      snippet: `A synthetic snapshot covering ${topic} with emphasis on actionable signals, confidence bands, and next-step recommendations.`,
      text: "",
    };
  });
}

function createFakeAnswer(domain: Domain): string {
  return `SemantikML synthesized answer for ${domain}: prioritize signal stability, monitor drift across cohorts, and validate retrieval quality using traceable evidence.`;
}

export default function Home() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const urlState = useMemo<UrlState>(() => {
    return {
      domain: parseDomain(searchParams.get("domain")),
      mode: parseMode(searchParams.get("mode")),
      query: searchParams.get("q") || "",
      seedId: searchParams.get("seed") || "",
      offset: parseNonNegativeInt(searchParams.get("offset"), 0),
      limit: parsePositiveInt(searchParams.get("limit"), 10),
      docId: searchParams.get("doc") || "",
      sidebarCollapsed: searchParams.get("sidebar") === "1",
      mock: searchParams.get("mock") === "1",
    };
  }, [searchParams]);

  const [draftQuery, setDraftQuery] = useState(urlState.query);
  const [draftLimit, setDraftLimit] = useState(String(urlState.limit));

  const [results, setResults] = useState<SearchResult[]>([]);
  const [answer, setAnswer] = useState<string>("");
  const [total, setTotal] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [ragStage, setRagStage] = useState("");
  const controllerRef = useRef<AbortController | null>(null);

  const [doc, setDoc] = useState<DocDetails | null>(null);
  const [docLoading, setDocLoading] = useState(false);
  const [docError, setDocError] = useState("");

  useEffect(() => {
    setDraftQuery(urlState.query);
  }, [urlState.query]);

  useEffect(() => {
    setDraftLimit(String(urlState.limit));
  }, [urlState.limit]);

  useEffect(() => {
    const params = new URLSearchParams(searchParams.toString());
    let needsUpdate = false;

    if (!params.get("domain")) {
      params.set("domain", urlState.domain);
      needsUpdate = true;
    }
    if (!params.get("mode")) {
      params.set("mode", urlState.mode);
      needsUpdate = true;
    }
    if (!params.get("offset")) {
      params.set("offset", String(urlState.offset));
      needsUpdate = true;
    }
    if (!params.get("limit")) {
      params.set("limit", String(urlState.limit));
      needsUpdate = true;
    }
    if (!params.get("sidebar")) {
      params.set("sidebar", urlState.sidebarCollapsed ? "1" : "0");
      needsUpdate = true;
    }

    if (needsUpdate) {
      router.replace(`?${params.toString()}`, { scroll: false });
    }
  }, [router, searchParams, urlState]);

  const updateParams = (next: Record<string, string | number | null | undefined>) => {
    const params = new URLSearchParams(searchParams.toString());
    Object.entries(next).forEach(([key, value]) => {
      if (value === undefined || value === null || value === "") {
        params.delete(key);
      } else {
        params.set(key, String(value));
      }
    });
    router.replace(`?${params.toString()}`, { scroll: false });
  };

  useEffect(() => {
    const controller = new AbortController();
    controllerRef.current = controller;
    const runSearch = async () => {
      setError("");
      setLoading(true);
      setResults([]);
      setAnswer("");
      setTotal(null);
      setRagStage(urlState.mode === "deep" ? "Retrieving context..." : "");

      if (urlState.mock) {
        const fakeResults = createFakeResults(urlState.domain, urlState.limit);
        setResults(fakeResults);
        setTotal(fakeResults.length + urlState.offset + 12);
        if (urlState.mode === "deep") {
          setAnswer(createFakeAnswer(urlState.domain));
        }
        setRagStage("");
        setLoading(false);
        return;
      }

      const needsQuery = urlState.mode !== "similar";
      if (needsQuery && !urlState.query) {
        setLoading(false);
        return;
      }
      if (urlState.mode === "similar" && !urlState.seedId) {
        setError("Seed id is required for Find Similar mode.");
        setLoading(false);
        return;
      }

      const endpoint =
        urlState.mode === "search"
          ? "/search"
          : urlState.mode === "deep"
          ? "/deep-research"
          : "/similar";

      const payload =
        urlState.mode === "similar"
          ? {
              seed_id: urlState.seedId,
              domain: urlState.domain,
              offset: urlState.offset,
              limit: urlState.limit,
            }
          : {
              query: urlState.query,
              domain: urlState.domain,
              offset: urlState.offset,
              limit: urlState.limit,
            };

      try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`Request failed (${response.status})`);
        }

        if (urlState.mode === "deep") {
          setRagStage("Synthesizing answer...");
        }
        const data = (await response.json()) as SearchResponse;
        setResults(data.results ?? []);
        setTotal(typeof data.total === "number" ? data.total : null);
        setAnswer(typeof data.answer === "string" ? data.answer : "");
      } catch (err) {
        if (err instanceof Error && err.name === "AbortError") {
          setRagStage("");
          return;
        }
        setError(err instanceof Error ? err.message : "Search failed.");
      } finally {
        setLoading(false);
        setRagStage("");
        controllerRef.current = null;
      }
    };

    runSearch();
    return () => controller.abort();
  }, [urlState]);

  useEffect(() => {
    if (!urlState.docId) {
      setDoc(null);
      setDocError("");
      return;
    }

    if (urlState.mock) {
      const match = results.find((result) => String(getDocId(result)) === urlState.docId);
      setDoc({
        id: urlState.docId,
        title: match?.title ?? "Synthetic result",
        domain: match?.domain ?? urlState.domain,
        source: match?.source ?? "SemantikML Lab",
        content:
          match?.snippet ??
          "Synthetic document preview generated for visual review. It highlights summary signals, risk factors, and suggested follow-up prompts.",
      });
      setDocLoading(false);
      setDocError("");
      return;
    }

    const controller = new AbortController();
    const loadDoc = async () => {
      setDocLoading(true);
      setDocError("");
      try {
        const response = await fetch(
          `${API_BASE}/doc/${encodeURIComponent(urlState.docId)}`,
          { signal: controller.signal }
        );
        if (!response.ok) {
          throw new Error(`Document lookup failed (${response.status})`);
        }
        const data = (await response.json()) as DocDetails;
        setDoc(data);
      } catch (err) {
        if (err instanceof Error && err.name === "AbortError") {
          return;
        }
        setDocError(err instanceof Error ? err.message : "Document lookup failed.");
      } finally {
        setDocLoading(false);
      }
    };

    loadDoc();
    return () => controller.abort();
  }, [results, urlState.docId, urlState.domain, urlState.mock]);

  const canGoBack = urlState.offset > 0;
  const canGoForward =
    total === null
      ? results.length === urlState.limit
      : urlState.offset + urlState.limit < total;
  const handleCancelRag = () => {
    if (controllerRef.current) {
      controllerRef.current.abort();
    }
  };

  return (
    <div className={styles.app}>
      <aside
        className={styles.sidebar}
        data-collapsed={urlState.sidebarCollapsed}
      >
        <div className={styles.sidebarHeader}>
          <div className={styles.signInWrap}>
            <span className={styles.signInDot} aria-hidden="true" />
            <a className={styles.signInButton} href="/sign-in">
              Sign in
            </a>
          </div>
          <button
            className={styles.collapseButton}
            type="button"
            onClick={() =>
              updateParams({ sidebar: urlState.sidebarCollapsed ? "0" : "1" })
            }
            aria-label={
              urlState.sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"
            }
          >
            {urlState.sidebarCollapsed ? ">>" : "<<"}
          </button>
        </div>
        <div className={styles.brandRow}>
          <div className={styles.brand}>SemantikML</div>
        </div>

        <div className={styles.controlBlock}>
          <div className={styles.controlLabel}>Domain</div>
          <div className={styles.segmented} role="group" aria-label="Domain">
            {DOMAINS.map((domain) => (
              <button
                key={domain}
                type="button"
                className={
                  domain === urlState.domain
                    ? styles.segmentedActive
                    : styles.segmentedButton
                }
                onClick={() =>
                  updateParams({ domain, offset: 0, doc: "" })
                }
              >
                {domain}
              </button>
            ))}
          </div>
        </div>

        <div className={styles.controlBlock}>
          <div className={styles.controlLabel}>Mode</div>
          <div className={styles.modeList} role="group" aria-label="Mode">
            {SIDEBAR_MODES.map((mode) => (
              <label key={mode.id} className={styles.modeOption}>
                <input
                  type="radio"
                  name="mode"
                  value={mode.id}
                  checked={urlState.mode === mode.id}
                  onChange={() =>
                    updateParams({
                      mode: mode.id,
                      offset: 0,
                      seed: "",
                      doc: "",
                    })
                  }
                />
                <span>{mode.label}</span>
              </label>
            ))}
          </div>
        </div>

        <div className={styles.controlBlock}>
          <div className={styles.controlLabel}>Pagination</div>
          <div className={styles.inlineRow}>
            <label className={styles.inlineLabel}>
              Limit
              <input
                className={styles.numberInput}
                value={draftLimit}
                onChange={(event) => setDraftLimit(event.target.value)}
                inputMode="numeric"
              />
            </label>
            <button
              className={styles.secondaryButton}
              type="button"
              onClick={() =>
                updateParams({
                  limit: parsePositiveInt(draftLimit, urlState.limit),
                  offset: 0,
                })
              }
            >
              Apply
            </button>
          </div>
        </div>

        <div className={styles.sidebarAccordion}>
          <details className={styles.accordionItem}>
            <summary className={styles.accordionSummary}>About</summary>
            <div className={styles.accordionBody}>
              <p>
                This project is a multi-domain semantic search engine that uses
                machine learning to improve how information is retrieved and
                ranked. The goal is to move beyond basic keyword matching and
                instead focus on understanding the meaning behind user queries.
              </p>
              <p>
                The system processes data from multiple domains and converts
                textual content into vector embeddings using modern language
                models. These embeddings allow the search engine to measure
                semantic similarity between queries and documents, producing
                more accurate and context-aware results.
              </p>
              <p>
                The backend is built in Python and handles dataset ingestion,
                cleaning, embedding generation, indexing, and search execution.
                It also supports different search modes, including fast semantic
                search and deeper retrieval workflows designed for more complex
                queries.
              </p>
              <p>
                The frontend acts as a bridge between the user and the
                underlying AI system. It allows users to interact with the
                search engine, change search settings, and view ranked results
                in a clear and user-friendly format.
              </p>
              <p>
                Overall, this project demonstrates how machine learning,
                backend engineering, and frontend design can be integrated into
                a single system to solve real-world information retrieval
                problems.
              </p>
            </div>
          </details>
          <details className={styles.accordionItem}>
            <summary className={styles.accordionSummary}>Pricing</summary>
            <div className={styles.accordionBody}>
              <p>Starter: $4/month for personal use and small projects.</p>
              <p>Team: $12/month per seat with shared workspaces.</p>
              <p>Scale: $29/month per seat with advanced controls.</p>
              <p>All plans include core semantic search and support.</p>
            </div>
          </details>
          <details className={styles.accordionItem}>
            <summary className={styles.accordionSummary}>Features</summary>
            <div className={styles.accordionBody}>
              <p>Search by meaning, not just keywords, for sharper relevance.</p>
              <p>Focus each query with clean, single-domain targeting.</p>
              <p>Deep Research answers grounded in your own source context.</p>
              <p>Find Similar surfaces stronger matches with smart re-ranking.</p>
              <p>Fast embedding pipeline ready for scale and future growth.</p>
            </div>
          </details>
        </div>

      </aside>

      <main className={styles.main}>
        <div className={styles.hero}>
          <div className={styles.heroEyebrow}>AI Search Engine</div>
          <h1 className={styles.heroTitle}>Find ML knowledge faster.</h1>
          <p className={styles.heroSubtitle}>
            Three domains. Three modes. One URL for every state.
          </p>
        </div>

        <form
          className={styles.searchBar}
          onSubmit={(event) => {
            event.preventDefault();
            updateParams({
              q: draftQuery,
              offset: 0,
              doc: "",
              seed: urlState.mode === "similar" ? urlState.seedId : "",
            });
          }}
        >
          <input
            className={styles.searchInput}
            value={draftQuery}
            onChange={(event) => setDraftQuery(event.target.value)}
            placeholder="Ask a question or search terms"
            disabled={urlState.mode === "similar"}
          />
          <button className={styles.primaryButton} type="submit">
            Run
          </button>
        </form>

        <div className={styles.statusRow}>
          <div className={styles.statusPill}>Domain: {urlState.domain}</div>
          <div className={styles.statusPill}>Mode: {urlState.mode}</div>
        </div>

        {loading && urlState.mode === "deep" ? (
          <div className={styles.ragStatus} role="status" aria-live="polite">
            <div className={styles.ragStatusLeft}>
              <span className={styles.ragSpinner} aria-hidden="true" />
              <span>{ragStage || "Working on deep research..."}</span>
            </div>
            <button
              type="button"
              className={styles.ragCancel}
              onClick={handleCancelRag}
            >
              Cancel
            </button>
          </div>
        ) : null}

        <section className={styles.resultsSection}>
          <div className={styles.resultsHeader}>
            <h2>Results</h2>
            <div className={styles.resultsMeta}>
              {loading
                ? "Loading..."
                : error
                ? "Error"
                : total !== null
                ? `${total} total`
                : `${results.length} shown`}
            </div>
          </div>

          {error ? <div className={styles.errorCard}>{error}</div> : null}

          {!loading && !error && urlState.mode === "deep" && answer ? (
            <article className={styles.answerCard}>
              <div className={styles.answerLabel}>Deep Research Answer</div>
              <p>{answer}</p>
            </article>
          ) : null}

          <div className={styles.resultsList}>
            {loading ? (
              <div className={styles.loadingCard}>Working...</div>
            ) : results.length === 0 ? (
              <div className={styles.emptyCard}>No results yet.</div>
            ) : (
              results.map((result, index) => {
                const docId = getDocId(result);
                const preview = getPreview(result);
                return (
                  <article
                    key={`${docId ?? "result"}-${index}`}
                    className={styles.resultCard}
                    style={{
                      animationDelay: `${index * 40}ms`,
                    }}
                  >
                    <div className={styles.resultHeader}>
                      <h3>{result.title || "Untitled"}</h3>
                      <div className={styles.scoreBadge}>
                        {typeof result.score === "number"
                          ? result.score.toFixed(4)
                          : "n/a"}
                      </div>
                    </div>
                    <div className={styles.resultMeta}>
                      <span>{result.domain || urlState.domain}</span>
                      <span>{result.source || "source"}</span>
                    </div>
                    {preview ? (
                      <p className={styles.resultPreview}>{preview}</p>
                    ) : null}
                    <div className={styles.resultActions}>
                      <button
                        type="button"
                        className={styles.ghostButton}
                        onClick={() =>
                          updateParams({ doc: docId ? String(docId) : "" })
                        }
                        disabled={!docId}
                      >
                        Open details
                      </button>
                    </div>
                  </article>
                );
              })
            )}
          </div>

          <div className={styles.pagination}>
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={() =>
                updateParams({
                  offset: Math.max(0, urlState.offset - urlState.limit),
                  doc: "",
                })
              }
              disabled={!canGoBack || loading}
            >
              Prev
            </button>
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={() =>
                updateParams({
                  offset: urlState.offset + urlState.limit,
                  doc: "",
                })
              }
              disabled={!canGoForward || loading}
            >
              Next
            </button>
          </div>
        </section>
      </main>

      <footer className={styles.footerPane}>
        <div className={styles.footerGrid}>
          <div className={styles.footerCard}>
            <div className={styles.footerLabel}>Live index</div>
            <div className={styles.footerKpi}>128k docs</div>
            <div className={styles.footerNote}>Updated 2 min ago</div>
          </div>
          <div className={styles.footerCard}>
            <div className={styles.footerLabel}>Query latency</div>
            <div className={styles.footerKpi}>~240ms</div>
            <div className={styles.footerNote}>Median on production</div>
          </div>
          <div className={styles.footerCard}>
            <div className={styles.footerLabel}>Confidence</div>
            <div className={styles.footerKpi}>92%</div>
            <div className={styles.footerNote}>Top-10 overlap</div>
          </div>
        </div>
        <div className={styles.footerBar}>
          <div className={styles.footerLinks}>
            <a className={styles.footerLink} href="#">
              Docs
            </a>
            <a className={styles.footerLink} href="#">
              API status
            </a>
            <a className={styles.footerLink} href="#">
              Support
            </a>
          </div>
          <button className={styles.footerCta} type="button">
            Request a demo
          </button>
        </div>
      </footer>

      <section
        className={styles.drawer}
        data-open={Boolean(urlState.docId)}
        aria-hidden={!urlState.docId}
      >
        <div className={styles.drawerHeader}>
          <h3>Document Details</h3>
          <div className={styles.drawerActions}>
            <button
              type="button"
              className={styles.similarButton}
              onClick={() =>
                updateParams({
                  mode: "similar",
                  seed: urlState.docId,
                  offset: 0,
                  doc: "",
                })
              }
              disabled={!urlState.docId}
            >
              Find similar
            </button>
            <button
              type="button"
              className={styles.closeButton}
              onClick={() => updateParams({ doc: "" })}
            >
              Close
            </button>
          </div>
        </div>
        <div className={styles.drawerBody}>
          {docLoading ? (
            <div className={styles.loadingCard}>Loading document...</div>
          ) : docError ? (
            <div className={styles.errorCard}>{docError}</div>
          ) : doc ? (
            <div className={styles.docContent}>
              <h4>{doc.title || "Untitled"}</h4>
              <div className={styles.docMeta}>
                <span>{doc.domain || urlState.domain}</span>
                <span>{doc.source || "source"}</span>
              </div>
              <p>{doc.content || doc.text || "No content available."}</p>
            </div>
          ) : (
            <div className={styles.emptyCard}>Select a document.</div>
          )}
        </div>
      </section>
    </div>
  );
}
