// Pure helpers for the Collection list (no React, no network).

/**
 * Next-page param for react-query's useInfiniteQuery.
 *
 * Must return `undefined` (not `false` or `null`) when there is no next
 * page: react-query v3 refetches every loaded page on invalidation and
 * only stops at `undefined`; anything else is sent as `cursor=...`.
 */
export const getNextPageParam = (lastPage) =>
  lastPage?.next_cursor ?? undefined;
