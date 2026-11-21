import {
  formsOf,
  getNextPageParam,
  patchRowInPages,
  usersOf,
} from "./labeledPaging";

describe("getNextPageParam", () => {
  it("returns undefined when next_cursor is null", () => {
    expect(getNextPageParam({ next_cursor: null })).toBe(undefined);
  });

  it("returns undefined when next_cursor is missing", () => {
    expect(getNextPageParam({})).toBe(undefined);
  });

  it("returns undefined when lastPage is undefined", () => {
    expect(getNextPageParam(undefined)).toBe(undefined);
  });

  it("returns the cursor when next_cursor is present", () => {
    expect(getNextPageParam({ next_cursor: "abc" })).toBe("abc");
  });

  it("never returns false", () => {
    expect(getNextPageParam({ next_cursor: null })).not.toBe(false);
    expect(getNextPageParam({})).not.toBe(false);
    expect(getNextPageParam(undefined)).not.toBe(false);
    expect(getNextPageParam({ next_cursor: "abc" })).not.toBe(false);
  });
});

const buildData = () => ({
  pageParams: [undefined, "c1"],
  pages: [
    {
      result: [
        { record_id: 1, state: { note: "a" } },
        { record_id: 2, state: {} },
      ],
      next_cursor: "c1",
      forms: { tags_form: ["t1"], lists_form: null },
      users: { 5: { name: "A" } },
    },
    {
      result: [{ record_id: 3, state: {}, extra: "keep" }],
      next_cursor: null,
      forms: { tags_form: ["t2"], lists_form: ["l"] },
      users: { 6: { name: "B" } },
    },
  ],
});

describe("patchRowInPages", () => {
  it("replaces the matching row and merges over the old row", () => {
    const data = buildData();
    const out = patchRowInPages(data, { record_id: 3, state: { note: "x" } });

    const patched = out.pages[1].result[0];
    expect(patched.state.note).toBe("x");
    expect(patched.extra).toBe("keep");
    // Unaffected page and rows are kept by reference.
    expect(out.pages[0]).toBe(data.pages[0]);
    expect(out.pages[0].result[0]).toBe(data.pages[0].result[0]);
    expect(out.pageParams).toBe(data.pageParams);
  });

  it("does not mutate the input", () => {
    const data = buildData();
    patchRowInPages(data, { record_id: 3, state: { note: "x" } });
    expect(data.pages[1].result[0].state.note).toBeUndefined();
    expect(data.pages[1].result[0].extra).toBe("keep");
  });

  it("returns the same object for an unknown id", () => {
    const data = buildData();
    expect(patchRowInPages(data, { record_id: 99 })).toBe(data);
  });
});

describe("formsOf", () => {
  it("returns the forms of the last loaded page", () => {
    expect(formsOf(buildData())).toEqual({
      tags_form: ["t2"],
      lists_form: ["l"],
    });
  });

  it("returns null placeholders when there are no pages", () => {
    expect(formsOf(undefined)).toEqual({
      tags_form: null,
      lists_form: null,
    });
    expect(formsOf({ pages: [] })).toEqual({
      tags_form: null,
      lists_form: null,
    });
  });
});

describe("usersOf", () => {
  it("merges users over all pages, later pages winning", () => {
    expect(usersOf(buildData())).toEqual({
      5: { name: "A" },
      6: { name: "B" },
    });
  });

  it("returns an empty object when there are no pages", () => {
    expect(usersOf(undefined)).toEqual({});
    expect(usersOf({ pages: [] })).toEqual({});
  });
});
