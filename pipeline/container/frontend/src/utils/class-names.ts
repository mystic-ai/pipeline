import { clsx, type ClassValue } from "clsx";
import {
  createTailwindMerge,
  getDefaultConfig,
  mergeConfigs,
} from "tailwind-merge";

export const cn = (...classes: ClassValue[]) => twMerge(clsx(...classes));

// Extends twMerge to allow for custom class groups
// IF you use the base twmerge function, some classes that start with `text-` will be removed accidentally

// ✖️ Normal twmerge: twMerge("font-bold text-gray-800 text-display_xs") => "font-bold text-display_xs"
// ✔️ New twmerge: twMerge("font-bold text-gray-800 text-display_xs") => "font-bold text-gray-800 text-display_xs"
export const twMerge = createTailwindMerge(() =>
  mergeConfigs(getDefaultConfig(), {
    classGroups: {
      text: [
        {
          text: [
            "smallest",
            "display_xs",
            "display_sm",
            "display_md",
            "display_lg",
            "display_xl",
            "display_2xl",
          ],
        },
      ],
    },
  })
);
