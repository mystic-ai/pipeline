"use client";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { useTheme } from "next-themes";
import { Button } from "../Buttons/Button";
import { IconMoon } from "../Icons/IconMoon";
import { IconSun } from "../Icons/IconSun";
import { useState } from "react";

export function ThemeDropdown(): JSX.Element {
  const { theme, setTheme } = useTheme();
  const [open, setOpen] = useState<boolean>(false);

  return (
    <DropdownMenu.Root onOpenChange={setOpen} open={open}>
      <DropdownMenu.Trigger asChild>
        <Button
          colorVariant="muted"
          aria-label="Choose theme"
          size="sm"
          active={open}
          className="w-[107px]"
        >
          <span className="flex gap-2">
            {theme === "dark" ? (
              <>
                <IconMoon size={16} />
                Dark mode
              </>
            ) : (
              <>
                <IconSun size={16} />
                Light mode
              </>
            )}
          </span>
        </Button>
      </DropdownMenu.Trigger>

      <DropdownMenu.Portal>
        <DropdownMenu.Content
          className="min-w-[120px] bg-white dark:bg-black border border-gray-300 dark:border-gray-700 rounded-md p-[5px] shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)] will-change-[opacity,transform] data-[side=top]:animate-slideDownAndFade data-[side=right]:animate-slideLeftAndFade data-[side=bottom]:animate-slideUpAndFade data-[side=left]:animate-slideRightAndFade z-50"
          sideOffset={5}
        >
          <DropdownMenu.Item className="select-none outline-none" asChild>
            <Button
              menuButton
              colorVariant="muted"
              onClick={() => setTheme("dark")}
            >
              Dark
            </Button>
          </DropdownMenu.Item>
          <DropdownMenu.Item className="select-none outline-none" asChild>
            <Button
              menuButton
              colorVariant="muted"
              onClick={() => setTheme("light")}
            >
              Light
            </Button>
          </DropdownMenu.Item>
          <DropdownMenu.Item className="select-none outline-none" asChild>
            <Button
              menuButton
              colorVariant="muted"
              onClick={() => setTheme("system")}
            >
              System
            </Button>
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
