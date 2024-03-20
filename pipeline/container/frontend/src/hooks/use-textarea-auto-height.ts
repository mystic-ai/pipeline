import { RefObject, useLayoutEffect } from "react";

interface UseTextareaAutoHeightProps {
  ref: RefObject<HTMLTextAreaElement>;
  autoHeight?: boolean;
}
export const useTextareaAutoHeight = ({
  ref,
  autoHeight,
}: UseTextareaAutoHeightProps) => {
  useLayoutEffect(() => {
    const listener = () => {
      if (ref.current) {
        // On change, set height to 0px
        ref.current.style.height = "0px";
        // Set height to scrollHeight
        ref.current.style.height = ref.current.scrollHeight + "px";
      }
    };
    if (autoHeight && ref.current) {
      // Trigger on load
      listener();

      // Disable scroll bar
      ref.current.style.overflow = "hidden";
      // Attach listener
      ref.current.addEventListener("input", listener);
    }

    return () => {
      autoHeight && ref.current
        ? ref.current.removeEventListener("input", listener)
        : null;
    };
  }, [ref, autoHeight]);
};
