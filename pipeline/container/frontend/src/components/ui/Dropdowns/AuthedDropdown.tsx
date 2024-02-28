"use client";

import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { UserMeProfileBox } from "@/components/sections/user/UserMeProfileBox";
import { Button } from "@/components/ui/Buttons/Button";
import { IconChevronDown } from "@/components/ui/Icons/IconChevronDown";
import { LineSeparator } from "@/components/ui/Separators/LineSeparator";
import { DescriptionText } from "@/components/ui/Typography/DescriptionText";
import { useGetMeUser } from "@/hooks/use-get-user";
import { routes } from "@/lib/routes";
import { LinkButton } from "../Buttons/LinkButton";
import { altRoutes, keyRoutes, subRoutes } from "./menu-items";
import { IconArrowTopRight } from "../Icons/IconArrowTopRight";
import { useState } from "react";

export const AuthedDropdown = () => {
  const { data: user } = useGetMeUser();
  const [open, setOpen] = useState<boolean>(false);

  // TO-DO: handle user loading/error
  return (
    <DropdownMenu.Root onOpenChange={setOpen} open={open}>
      <DropdownMenu.Trigger asChild>
        <Button
          colorVariant="muted"
          aria-label="open the account dropdown for more links"
          size="md"
          active={open}
          className="
            lg:h-auto py-[.3125rem] px-[.3125rem]
            [&>svg]:data-[state=open]:rotate-180
          "
        >
          <UserMeProfileBox />

          <IconChevronDown
            size={20}
            className={`
              transition
            `}
            aria-hidden="true"
          />
        </Button>
      </DropdownMenu.Trigger>

      <DropdownMenu.Portal>
        <DropdownMenu.Content
          className="min-w-[120px] bg-white dark:bg-black border border-gray-300 dark:border-gray-700 rounded-md p-[5px] shadow-[0px_10px_38px_-10px_rgba(22,_23,_24,_0.35),_0px_10px_20px_-15px_rgba(22,_23,_24,_0.2)] will-change-[opacity,transform] data-[side=top]:animate-slideDownAndFade data-[side=right]:animate-slideLeftAndFade data-[side=bottom]:animate-slideUpAndFade data-[side=left]:animate-slideRightAndFade z-50"
          sideOffset={5}
          align="end"
        >
          <DropdownMenu.Item className="select-none outline-none" asChild>
            <LinkButton
              href={routes.products.username(user?.username || "").home.href}
              aria-label={
                routes.products.username(user?.username || "").home.ariaLabel
              }
              className="w-full block rounded"
              buttonProps={{
                className: "flex-col !gap-0 items-start",
                size: "custom",
                menuButton: true,
                colorVariant: "muted",
              }}
            >
              <p className="text-base font-semibold w-full truncate">
                {user?.username}
              </p>
              <DescriptionText>View my public page</DescriptionText>
            </LinkButton>
          </DropdownMenu.Item>

          <DropdownMenu.Separator className="-mx-[5px] py-[5px]">
            <LineSeparator />
          </DropdownMenu.Separator>

          {keyRoutes.map((p, index) => {
            return (
              <DropdownMenu.Item
                key={p.title}
                className="select-none outline-none sm2:hidden"
                asChild
              >
                <LinkButton
                  href={p.route?.href}
                  aria-label={p.route?.ariaLabel}
                  className="w-full block rounded"
                  buttonProps={{
                    size: "sm",
                    menuButton: true,
                    colorVariant: "muted",
                  }}
                >
                  {p.title}
                </LinkButton>
              </DropdownMenu.Item>
            );
          })}

          {subRoutes.map((p, index) => {
            return (
              <DropdownMenu.Item
                className="select-none outline-none"
                asChild
                key={p.title}
              >
                <LinkButton
                  href={p.route?.href}
                  aria-label={p.route?.ariaLabel}
                  className="w-full block rounded"
                  buttonProps={{
                    size: "sm",
                    menuButton: true,
                    colorVariant: "muted",
                  }}
                >
                  {p.title}
                </LinkButton>
              </DropdownMenu.Item>
            );
          })}

          <DropdownMenu.Separator className="-mx-[5px] py-[5px]">
            <LineSeparator />
          </DropdownMenu.Separator>

          {altRoutes.map((p, index) => {
            return (
              <DropdownMenu.Item
                className="select-none outline-none"
                asChild
                key={p.title}
              >
                <LinkButton
                  href={p.route?.href}
                  aria-label={p.route?.ariaLabel}
                  className="w-full block rounded"
                  buttonProps={{
                    size: "sm",
                    menuButton: true,
                    colorVariant: "muted",
                  }}
                  target="_blank"
                >
                  {p.title}
                  <IconArrowTopRight className="stroke-gray-700" />
                </LinkButton>
              </DropdownMenu.Item>
            );
          })}

          <DropdownMenu.Separator className="-mx-[5px] py-[5px]">
            <LineSeparator />
          </DropdownMenu.Separator>

          <DropdownMenu.Item className="select-none outline-none" asChild>
            <LinkButton
              href={routes.products.signout.href}
              aria-label={routes.products.signout.ariaLabel}
              className="w-full block rounded"
              buttonProps={{
                size: "sm",
                menuButton: true,
                colorVariant: "muted",
              }}
            >
              Logout
            </LinkButton>
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
};
