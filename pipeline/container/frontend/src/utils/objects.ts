/**
 * Transform normal element.getBoundingClientRect() into a normal JS object
 *
 * @param {Element} element
 * @return {*}
 */
export const getBoundingClientRect = (element: Element) => {
  const { top, right, bottom, left, width, height, x, y } =
    element.getBoundingClientRect();
  return { top, right, bottom, left, width, height, x, y };
};

export const isObjectEmpty = (obj: object) => {
  for (const prop in obj) {
    if (Object.hasOwn(obj, prop)) {
      return false;
    }
  }

  return true;
};

export const isObject = (value: any): boolean => {
  try {
    return (
      typeof value === "object" && value !== null && !(value instanceof Array)
    );
  } catch (e) {
    return false;
  }
};
