const { describe } = require("node:test");
const { foo } = require("./foo");

describe("foo", () => {
  it('should return "hello world"', () => {
    expect(foo()).toBe("hello world");
  });
});
