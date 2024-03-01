import { cva, type VariantProps } from "class-variance-authority";
import { twMerge } from "@/lib/helpers.className";

const badge = cva(
  "w-[1.1215rem] h-[1.1215rem] rounded-xl text-xs font-medium",
  {
    variants: {
      variant: {
        default: ["bg-gray-100", "text-gray-700", "dark:bg-black"],
        error: [
          "ring-1",
          "bg-error-50",
          "ring-error-300",
          "text-error-700",
          "dark:bg-black",
          "dark:ring-error-400",
          "dark:text-white",
        ],
        success: [
          "ring-1",
          "bg-success-50",
          "dark:bg-black",
          "ring-success-700",
          "dark:ring-success-400",
          "text-success-700",
          "dark:text-success-400",
        ],
      },
      defaultVariants: {
        variant: "default",
      },
    },
  }
);

interface Props extends VariantProps<typeof badge> {
  text: string;
  variant: "default" | "error" | "success";
}

export function CircleBadge({ text, variant = "default" }: Props): JSX.Element {
  return <div className={twMerge(badge({ variant }))}>{text}</div>;
}
