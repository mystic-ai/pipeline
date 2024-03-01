import { v4 as uuidv4 } from "uuid";

export const range = (start: number, end: number) =>
  Array.from(Array(Math.abs(end - start) + 1), (_, i) => start + i);

export const isArray = (value: any): boolean => {
  try {
    return Array.isArray(value);
  } catch (e) {
    return false;
  }
};

export function assignUUIDsToData(data: any[]): any[] {
  return data.map((item) => ({
    ...item,
    uuid: uuidv4(),
  }));
}
