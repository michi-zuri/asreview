import React from "react";
import { useQueryClient } from "react-query";

import { ProjectAPI } from "api";
import {
  appendPage,
  applyForms,
  applyRowUpdates,
  emptyState,
  markChanges,
  reconcile,
  snapshotParams,
} from "./collectionModel";

// Spec 0001 Part 4: the Collection's snapshot state machine, wired to the
// network. The logic itself lives in collectionModel.js (pure, tested); this
// hook only loads pages, polls `labeled/changes`, patches cards via
// `records?ids=`, and runs Refresh = `labeled/ids` → reconcile → `records?ids=`.
// Nothing refetches a page.

export const POLL_MS = 15000;
export const PATCH_CHUNK = 200;

export default function useCollection({ project_id, subset, filter }) {
  const queryClient = useQueryClient();

  const [state, setState] = React.useState(() => emptyState(subset));
  const [isLoading, setIsLoading] = React.useState(true);
  const [isLoadingMore, setIsLoadingMore] = React.useState(false);
  const [error, setError] = React.useState(null);

  const stateRef = React.useRef(state);
  const gen = React.useRef(0);
  const busy = React.useRef(false);
  const wantMore = React.useRef(false);
  const wantRefresh = React.useRef(false);
  const dirtyIds = React.useRef(new Set());

  const snapshotKey = JSON.stringify([project_id, subset, filter]);

  const commit = React.useCallback((next) => {
    stateRef.current = next;
    setState(next);
  }, []);

  // Fetch rows for a list of ids in PATCH_CHUNK-sized chunks, patching each
  // chunk in place as it arrives (so a partial patch is already visible).
  // `g` is the snapshot generation; a snapshot change drops the result.
  const patchIds = React.useCallback(
    async (ids, g) => {
      for (let i = 0; i < ids.length; i += PATCH_CHUNK) {
        if (gen.current !== g) return;
        const chunk = ids.slice(i, i + PATCH_CHUNK);
        const resp = await ProjectAPI.fetchRecordsByIds({
          project_id,
          ids: chunk,
        });
        if (gen.current !== g) return;
        commit(
          applyRowUpdates(
            stateRef.current,
            resp.result || [],
            resp.users || {},
          ),
        );
      }
    },
    [project_id, commit],
  );

  // config_version changed: refetch forms + users once (`/labeled?per_page=1`);
  // the single row is ignored.
  const applyFormsOnce = React.useCallback(
    async (g) => {
      const page = await ProjectAPI.fetchLabeledPage({
        project_id,
        subset,
        filter,
        per_page: 1,
      });
      if (gen.current !== g) return;
      commit(applyForms(stateRef.current, page));
    },
    [project_id, subset, filter, commit],
  );

  // A fresh snapshot: empty state, then the first page. Used on mount, on a
  // filter/subset change, and by Refresh after the server's version counter
  // went backwards (state.rewound).
  const startSnapshot = React.useCallback(() => {
    gen.current += 1;
    const g = gen.current;
    commit(emptyState(subset));
    dirtyIds.current.clear();
    setIsLoading(true);
    ProjectAPI.fetchLabeledPage({ project_id, subset, filter, per_page: 50 })
      .then((page) => {
        if (gen.current !== g) return;
        commit(appendPage(stateRef.current, page));
        setError(null);
        setIsLoading(false);
      })
      .catch((err) => {
        if (gen.current !== g) return;
        setError(err?.message || "Failed to load");
        setIsLoading(false);
      });
  }, [project_id, subset, filter, commit]);

  // Refs so an operation can start one that was queued while it ran, with
  // the latest closure.
  const loadMoreRef = React.useRef(() => {});
  const refreshRef = React.useRef(() => {});

  // Called when an operation ends: release the lock and run what was queued
  // meanwhile (a Refresh click first, then a pending loadMore).
  const finish = React.useCallback(() => {
    busy.current = false;
    if (wantRefresh.current) {
      wantRefresh.current = false;
      refreshRef.current();
    } else if (wantMore.current) {
      wantMore.current = false;
      loadMoreRef.current();
    }
  }, []);

  const loadMore = React.useCallback(() => {
    if (!stateRef.current.hasMore) return;
    if (busy.current) {
      wantMore.current = true;
      return;
    }
    busy.current = true;
    setIsLoadingMore(true);
    const g = gen.current;
    const boundary = stateRef.current.boundary;
    const cursor = boundary && boundary !== "end" ? boundary : undefined;
    ProjectAPI.fetchLabeledPage({
      project_id,
      subset,
      filter,
      cursor,
      per_page: 50,
    })
      .then((page) => {
        if (gen.current !== g) return;
        commit(appendPage(stateRef.current, page));
        setError(null);
      })
      .catch((err) => {
        if (gen.current !== g) return;
        setError(err?.message || "Failed to load more");
      })
      .finally(() => {
        setIsLoadingMore(false);
        finish();
      });
  }, [project_id, subset, filter, commit, finish]);
  loadMoreRef.current = loadMore;

  const refresh = React.useCallback(() => {
    if (busy.current) {
      // Never drop a click: run as soon as the current operation ends.
      wantRefresh.current = true;
      return;
    }
    if (stateRef.current.rewound) {
      // The server's versions went backwards; ours can't be compared with
      // them any more, so reconciling could keep outdated rows.
      startSnapshot();
      return;
    }
    busy.current = true;
    const g = gen.current;
    const prevConfigVersion = stateRef.current.configVersion;
    const boundary = snapshotParams(stateRef.current).boundary;
    ProjectAPI.fetchLabeledIds({ project_id, subset, filter, boundary })
      .then((resp) => {
        if (gen.current !== g) return;
        const {
          state: next,
          idsToFetch,
          configChanged,
        } = reconcile(stateRef.current, resp, Array.from(dirtyIds.current));
        commit(next);
        queryClient.invalidateQueries(["fetchLabeledStats", { project_id }]);
        const tail = idsToFetch.length
          ? patchIds(idsToFetch, g)
          : Promise.resolve();
        return tail.then(() => (configChanged ? applyFormsOnce(g) : undefined));
      })
      .then(() => {
        if (gen.current !== g) return;
        setError(null);
      })
      .catch((err) => {
        if (gen.current !== g) return;
        setError(err?.message || "Failed to refresh");
        // reconcile may already have cleared `stale` and dropped the rows the
        // failed fetch was to bring. Keep the banner so Refresh can be retried:
        // the next reconcile fetches exactly what is still missing or old.
        commit({
          ...stateRef.current,
          stale: true,
          configVersion: prevConfigVersion,
        });
      })
      .finally(finish);
  }, [
    project_id,
    subset,
    filter,
    commit,
    queryClient,
    patchIds,
    applyFormsOnce,
    startSnapshot,
    finish,
  ]);
  refreshRef.current = refresh;

  const poll = React.useCallback(() => {
    if (document.hidden) return;
    if (busy.current) return; // skipped; the next tick retries
    if (stateRef.current.watermark == null) return; // nothing loaded yet
    busy.current = true;
    const g = gen.current;
    const since = stateRef.current.watermark;
    const prevConfigVersion = stateRef.current.configVersion;
    const boundary = snapshotParams(stateRef.current).boundary;
    ProjectAPI.fetchLabeledChanges({
      project_id,
      subset,
      filter,
      since,
      boundary,
    })
      .then((resp) => {
        if (gen.current !== g) return;
        const {
          state: next,
          idsToPatch,
          configChanged,
        } = markChanges(stateRef.current, resp);
        commit(next);
        if (resp.reset || (resp.changes && resp.changes.length > 0)) {
          queryClient.invalidateQueries(["fetchLabeledStats", { project_id }]);
        }
        const tail = idsToPatch.length
          ? patchIds(idsToPatch, g)
          : Promise.resolve();
        return tail.then(() => (configChanged ? applyFormsOnce(g) : undefined));
      })
      .then(() => {
        if (gen.current !== g) return;
        setError(null);
      })
      .catch(() => {
        // A failed poll shows nothing; the next tick retries. markChanges may
        // already have moved the watermark and configVersion past changes
        // whose patch or forms fetch failed: step back so the next poll
        // reports them again (a re-report is harmless, a lost patch is not).
        if (gen.current !== g) return;
        const s = stateRef.current;
        if (s.watermark > since || s.configVersion !== prevConfigVersion) {
          commit({ ...s, watermark: since, configVersion: prevConfigVersion });
        }
      })
      .finally(finish);
  }, [
    project_id,
    subset,
    filter,
    commit,
    queryClient,
    patchIds,
    applyFormsOnce,
    finish,
  ]);

  // A fresh snapshot on mount and on every filter/subset change.
  React.useEffect(() => {
    startSnapshot();
    return () => {
      gen.current += 1;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [snapshotKey]);

  // Poll every POLL_MS, and once immediately when the tab becomes visible.
  React.useEffect(() => {
    const timer = setInterval(poll, POLL_MS);
    const onVisibility = () => {
      if (!document.hidden) poll();
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [poll]);

  const onRowUpdated = React.useCallback(
    (row) => {
      if (!row) return;
      commit(applyRowUpdates(stateRef.current, [row]));
      queryClient.invalidateQueries(["fetchLabeledStats", { project_id }]);
    },
    [commit, queryClient, project_id],
  );

  const onEditingChange = React.useCallback((record_id, editing) => {
    if (editing) dirtyIds.current.add(record_id);
    else dirtyIds.current.delete(record_id);
  }, []);

  return {
    state,
    isLoading,
    isLoadingMore,
    error,
    loadMore,
    refresh,
    onRowUpdated,
    onEditingChange,
  };
}
