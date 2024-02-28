"use client";
import * as NavigationMenu from "@radix-ui/react-navigation-menu";
import { useTheme } from "next-themes";
import { Button } from "../Buttons/Button";
import { IconMoon } from "../Icons/IconMoon";
import { IconSun } from "../Icons/IconSun";
import { useState } from "react";
import { LinkButton } from "../Buttons/LinkButton";
import { routes } from "@/lib/routes";
import { IconChevronDown } from "../Icons/IconChevronDown";

export function UseCasesDropdown(): JSX.Element {
  const [openMenu, setOpenMenu] = useState<string>();

  let dropdownName = "dropdown-1";

  return (
    <NavigationMenu.Root
      className="relative"
      delayDuration={0}
      onValueChange={setOpenMenu}
    >
      <NavigationMenu.List className="">
        <NavigationMenu.Item value={dropdownName}>
          <NavigationMenu.Trigger asChild>
            <Button
              colorVariant="link"
              darkmode
              aria-label="Choose theme"
              active={openMenu ? true : false}
              className="pr-1"
            >
              Use cases
              <IconChevronDown
                className={`!stroke-gray-500 transition ${
                  openMenu ? "rotate-180" : ""
                }`}
              />
            </Button>
          </NavigationMenu.Trigger>

          <NavigationMenu.Content
            className="
              bg-white dark:bg-black border border-gray-300 dark:border-gray-700
              rounded-md
              p-[5px]
              shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)]
              will-change-[opacity,transform] data-[side=top]:animate-slideDownAndFade
              z-50
              data-[motion=from-start]:animate-enterFromLeft
              data-[motion=from-end]:animate-enterFromRight
              data-[motion=to-start]:animate-exitToLeft
              data-[motion=to-end]:animate-exitToRight

            "
          >
            <ul className="">
              <li>
                <NavigationMenu.Link asChild>
                  <LinkButton
                    href={routes.general.useCases.text.href}
                    aria-label={routes.general.useCases.text.ariaLabel}
                    className="w-full"
                    buttonProps={{
                      colorVariant: "muted",
                      menuButton: true,
                    }}
                  >
                    Text
                  </LinkButton>
                </NavigationMenu.Link>
              </li>
              <li>
                <NavigationMenu.Link asChild>
                  <LinkButton
                    href={routes.general.useCases.imageVideo.href}
                    aria-label={routes.general.useCases.imageVideo.ariaLabel}
                    className="w-full"
                    buttonProps={{
                      colorVariant: "muted",
                      menuButton: true,
                    }}
                  >
                    Image and video
                  </LinkButton>
                </NavigationMenu.Link>
              </li>
              <li>
                <NavigationMenu.Link asChild>
                  <LinkButton
                    href={routes.general.useCases.audio.href}
                    aria-label={routes.general.useCases.audio.ariaLabel}
                    className="w-full"
                    buttonProps={{
                      colorVariant: "muted",
                      menuButton: true,
                    }}
                  >
                    Audio
                  </LinkButton>
                </NavigationMenu.Link>
              </li>
            </ul>
          </NavigationMenu.Content>
        </NavigationMenu.Item>

        <NavigationMenu.Indicator
          className="
            data-[state=visible]:animate-fadeIn
            data-[state=hidden]:animate-fadeOut
            top-full
            z-[1]
            flex
            h-[10px]
            items-end
            justify-center
            overflow-hidden
            transition-[width,transform_250ms_ease]
          "
        >
          <div className="relative top-[70%] h-[10px] w-[10px] rotate-[45deg] rounded-tl-[2px] bg-white dark:bg-gray-700" />
        </NavigationMenu.Indicator>
      </NavigationMenu.List>

      <div className="absolute top-full left-0 flex w-full justify-center">
        <NavigationMenu.Viewport
          className="
            data-[state=open]:animate-scaleIn
            data-[state=closed]:animate-scaleOut
            relative
            mt-[10px]
            origin-[top_center]
            rounded-[6px]
            bg-white dark:bg-gray-900
            transition-[width,_height]
            duration-300
            h-[var(--radix-navigation-menu-viewport-height)]
            w-full
            sm:w-[var(--radix-navigation-menu-viewport-width)]
          "
        />
      </div>
    </NavigationMenu.Root>
  );
}
