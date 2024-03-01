import React from "react";
import { SyntheticEvent, useState } from "react";
import { Button, type ButtonSize } from "./Button";
import { IconRefresh } from "../Icons/IconRefresh";

interface Props {
  size: ButtonSize;
  className?: string;
  onClick: (e: SyntheticEvent) => void;
}

export function RefreshButton({
  size = "xxs",
  className = "",
  onClick,
}: Props) {
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);

  function handleClick(e: any) {
    setIsRefreshing(true);
    onClick && onClick(e);
    setTimeout(() => {
      setIsRefreshing(false);
    }, 1000);
  }

  return (
    <Button
      size={size}
      colorVariant="secondary"
      onClick={handleClick}
      className={`${className}`}
    >
      <IconRefresh
        className={`origin-center ${isRefreshing ? "animate-spin" : ""}`}
      />
      Refresh
    </Button>
  );
}
