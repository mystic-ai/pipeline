"use client";

import { Button } from "@/components/ui/Buttons/Button";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { useState } from "react";
import { LinkButton } from "../Buttons/LinkButton";
import { companyItems, resourcesItems, offeringsItems, useCaseItems } from "./menu-items";
import { IconArrowTopRight } from "../Icons/IconArrowTopRight";

export const MobileDropdown = () => {
  const [open, setOpen] = useState<boolean>(false);

  return (
    <DropdownMenu.Root onOpenChange={setOpen} open={open}>
      <DropdownMenu.Trigger className="mobile-dropdown-trigger" asChild>
        <Button
          colorVariant="muted"
          justIcon
          aria-label="Choose theme"
          size="sm"
          active={open}
        >
          <div
            className={`
              animated-menu-icon
              ${open ? "open" : ""}
            `}
          >
            <span></span>
            <span></span>
            <span></span>
          </div>
        </Button>
      </DropdownMenu.Trigger>

      <DropdownMenu.Portal>
        <DropdownMenu.Content
          className="dark
          min-w-[220px]
          bg-black
          border border-gray-700
          rounded-md
          p-[5px]
          will-change-[opacity,transform]
          data-[side=top]:animate-slideDownAndFade
          data-[side=bottom]:animate-slideUpAndFade
          z-50
          flex flex-col gap-4 right-0 absolute"
          sideOffset={5}
          align="end"
        >
          <DropdownMenu.Group>
            <DropdownMenu.Label asChild>
              <p className="text-xs font-medium text-gray-400 py-1 px-2">
                Offerings
              </p>
            </DropdownMenu.Label>

            {offeringsItems.map((p, index) => {
              return (
                <DropdownMenu.Item
                  className="select-none outline-none relative data-[highlighted]:z-10"
                  asChild
                  key={p.href}
                >
                  <LinkButton
                    key={p.href}
                    href={p.href}
                    rel={p.rel}
                    target={p.target ?? p.target}
                    aria-label={p.ariaLabel}
                    className="w-full"
                    buttonProps={{
                      colorVariant: "link",
                      darkmode: true,
                      size: "sm",
                      menuButton: true,
                    }}
                  >
                    {p.title}
                    {p.target === "_blank" ? (
                      <IconArrowTopRight className="stroke-gray-700" />
                    ) : null}
                  </LinkButton>
                </DropdownMenu.Item>
              );
            })}
          </DropdownMenu.Group>
          <DropdownMenu.Group>
            <DropdownMenu.Label asChild>
              <p className="text-xs font-medium text-gray-400 py-1 px-2">
                Use cases
              </p>
            </DropdownMenu.Label>

            {useCaseItems.map((p, index) => {
              return (
                <DropdownMenu.Item
                  className="select-none outline-none relative data-[highlighted]:z-10"
                  asChild
                  key={p.href}
                >
                  <LinkButton
                    key={p.href}
                    href={p.href}
                    rel={p.rel}
                    target={p.target ?? p.target}
                    aria-label={p.ariaLabel}
                    className="w-full"
                    buttonProps={{
                      colorVariant: "link",
                      darkmode: true,
                      size: "sm",
                      menuButton: true,
                    }}
                  >
                    {p.title}
                    {p.target === "_blank" ? (
                      <IconArrowTopRight className="stroke-gray-700" />
                    ) : null}
                  </LinkButton>
                </DropdownMenu.Item>
              );
            })}
          </DropdownMenu.Group>


          <DropdownMenu.Group>
            <DropdownMenu.Label asChild>
              <p className="text-xs font-medium text-gray-400 py-1 px-2">
                Resources
              </p>
            </DropdownMenu.Label>

            {resourcesItems.map((p, index) => {
              return (
                <DropdownMenu.Item
                  className="select-none outline-none relative data-[highlighted]:z-10"
                  asChild
                  key={p.href}
                >
                  <LinkButton
                    key={p.href}
                    href={p.href}
                    rel={p.rel}
                    target={p.target ?? p.target}
                    aria-label={p.ariaLabel}
                    className="w-full"
                    buttonProps={{
                      colorVariant: "link",
                      darkmode: true,
                      size: "sm",
                      menuButton: true,
                    }}
                  >
                    {p.title}
                    {p.target === "_blank" ? (
                      <IconArrowTopRight className="stroke-gray-700" />
                    ) : null}
                  </LinkButton>
                </DropdownMenu.Item>
              );
            })}
          </DropdownMenu.Group>

          <DropdownMenu.Group>
            <DropdownMenu.Label asChild>
              <p className="text-xs font-medium text-gray-400 py-1 px-2">
                Company
              </p>
            </DropdownMenu.Label>

            {companyItems.map((p, index) => {
              return (
                <DropdownMenu.Item
                  className="select-none outline-none relative data-[highlighted]:z-10"
                  asChild
                  key={p.href}
                >
                  <LinkButton
                    key={p.href}
                    href={p.href}
                    rel={p.rel}
                    target={p.target ?? p.target}
                    aria-label={p.ariaLabel}
                    className="w-full"
                    buttonProps={{
                      colorVariant: "link",
                      darkmode: true,
                      size: "sm",
                      menuButton: true,
                    }}
                  >
                    {p.title}
                    {p.target === "_blank" ? (
                      <IconArrowTopRight className="stroke-gray-700" />
                    ) : null}
                  </LinkButton>
                </DropdownMenu.Item>
              );
            })}
          </DropdownMenu.Group>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
};
