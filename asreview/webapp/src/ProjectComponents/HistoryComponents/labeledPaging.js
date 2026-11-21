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

/**
 * Return a copy of react-query infinite data with the row whose
 * record_id equals row.record_id replaced by `row` (merged over the
 * old row, so client-only fields survive). Pages and other rows
 * are kept by reference. Unknown id → data unchanged (same object).
 */
export const patchRowInPages = (data, row) => {
  if (!data || !row) return data;
  let patched = false;
  const pages = data.pages.map((page) => {
    if (!page.result.some((r) => r.record_id === row.record_id)) {
      return page;
    }
    patched = true;
    return {
      ...page,
      result: page.result.map((r) =>
        r.record_id === row.record_id ? { ...r, ...row } : r,
      ),
    };
  });
  return patched ? { ...data, pages } : data;
};

/**
 * Forms from the LAST loaded page (the most recent server answer).
 * Returns null placeholders when there are no pages or no forms.
 */
export const formsOf = (data) => {
  const pages = data?.pages;
  if (!pages || pages.length === 0) {
    return { tags_form: null, lists_form: null };
  }
  const last = pages[pages.length - 1];
  return {
    tags_form: last.forms?.tags_form ?? null,
    lists_form: last.forms?.lists_form ?? null,
  };
};

/**
 * Users merged over all pages (later pages win); `{}` when no pages.
 */
export const usersOf = (data) => {
  const pages = data?.pages;
  if (!pages || pages.length === 0) return {};
  return pages.reduce((acc, page) => ({ ...acc, ...(page.users || {}) }), {});
};
