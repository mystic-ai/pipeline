import React from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { PropsWithChildren } from "react";
import { IconWrapper } from "../../ui/Icons/IconWrapper";
import { IconSettings4 } from "../../ui/Icons/IconSettings4";

interface Props extends PropsWithChildren {
  isOpen: boolean;
  handleClose: () => void;
}

export function ChatAppSettingsDialog({ handleClose, children }: Props) {
  return (
    <Dialog.Root defaultOpen={true} onOpenChange={(open) => handleClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="dialog-content--side-panel">
          <div className="dialog-header">
            <IconWrapper variant={"default"}>
              <IconSettings4 />
            </IconWrapper>

            <div className="dialog-header-texts">
              <Dialog.Title className="dialog-title">Settings</Dialog.Title>
              <Dialog.Description className="dialog-description">
                Edit the settings of the chat app.
              </Dialog.Description>
            </div>
          </div>

          {children}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
