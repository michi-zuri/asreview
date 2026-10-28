import { getNextPageParam } from "./labeledPaging";

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
