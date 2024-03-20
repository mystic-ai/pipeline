import React from "react";

export interface Props {
  children: React.ReactNode;
  isClickable?: boolean;
  isModal?: boolean;
  className?: string;
  ariaLabel: string;
  handleClick?: () => void;
}

export function ButtonCard({
  children = "",
  isClickable = false,
  isModal = false,
  className = "",
  handleClick,
  ariaLabel = "",
}: Props) {
  const styles = isModal ? "shadow-lg" : "";
  const hoverStyles = isClickable ? "hover:bg-gray100 hover:border-gray3" : "";

  return (
    <button
      onClick={() => handleClick && handleClick()}
      className={`border rounded-lg border-gray200 bg-white p-4 ${styles} ${hoverStyles} ${className}`}
      aria-label={ariaLabel}
    >
      {children}
    </button>
  );
}
