import React, {
  useState,
  useCallback,
  useMemo,
  createContext,
  PropsWithChildren,
  ReactNode,
} from "react";
import * as ToastPrimitive from "@radix-ui/react-toast";
import { AnimatePresence, motion } from "framer-motion";
import { cva } from "class-variance-authority";
import { Button } from "../Buttons/Button";
import { IconHeart } from "../Icons/IconHeart";
import { DescriptionText } from "../Typography/DescriptionText";
import { twMerge } from "../../../utils/class-names";
import { IconWrapper } from "../Icons/IconWrapper";
import { IconAlertCircle } from "../Icons/IconAlertCircle";
import { IconCheckmarkCircle } from "../Icons/IconCheckmarkCircle";
import { IconX } from "../Icons/IconX";

const notificationVariants = cva(
  `relative w-full gap-5 pointer-events-auto overflow-hidden
   bg-white dark:bg-gray-950
   rounded-md
   border border-gray-300 dark:border-gray-700
   pl-6 pr-4 py-4 shadow-lg
   `,
  {
    variants: {
      variant: {
        default: "bg-white",
        error: "group",
        success: "group",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

interface Props extends PropsWithChildren {
  position?: "topLeft" | "topRight" | "bottomLeft" | "bottomRight";
}

interface NotificationProps {
  variant?: "default" | "success" | "error";
  title: string;
  description?: ReactNode | string;
  duration?: number;
}

const NotificationContext = createContext({
  default: ({ title, description, duration }: NotificationProps) => {},
  success: ({ title, description, duration }: NotificationProps) => {},
  error: ({ title, description, duration }: NotificationProps) => {},
});

const Notifications = ({ position, children }: Props) => {
  const [notifications, setNotifications] = useState(new Map());
  const isPositionedTop = position === "topLeft" || position === "topRight";

  const handleAddToast = useCallback((toast: NotificationProps) => {
    setNotifications((prev) => {
      const newMap = new Map(prev);
      newMap.set(String(Date.now()), { ...toast });
      return newMap;
    });
  }, []);

  const handleRemoveToast = useCallback((key: NotificationProps) => {
    setNotifications((prev) => {
      const newMap = new Map(prev);
      newMap.delete(key);
      return newMap;
    });
  }, []);

  const handleDispatchDefault = useCallback(
    ({ title, description, duration = 5000 }: NotificationProps) =>
      handleAddToast({ title, description, duration, variant: "default" }),
    [handleAddToast]
  );

  const handleDispatchSuccess = useCallback(
    ({ title, description, duration = 5000 }: NotificationProps) =>
      handleAddToast({ title, description, duration, variant: "success" }),
    [handleAddToast]
  );

  const handleDispatchError = useCallback(
    ({ title, description, duration = 10000 }: NotificationProps) =>
      handleAddToast({ title, description, duration, variant: "error" }),
    [handleAddToast]
  );

  return (
    <NotificationContext.Provider
      value={useMemo(
        () => ({
          default: handleDispatchDefault,
          success: handleDispatchSuccess,
          error: handleDispatchError,
        }),
        [handleDispatchDefault, handleDispatchSuccess, handleDispatchError]
      )}
    >
      <ToastPrimitive.Provider swipeDirection="right">
        {children}
        <AnimatePresence>
          {Array.from(notifications).map(([key, notification]) => {
            return (
              <ToastPrimitive.Root
                className="data-[state=open]:animate-slideIn data-[state=closed]:animate-hide data-[swipe=move]:translate-x-[var(--radix-toast-swipe-move-x)] data-[swipe=cancel]:translate-x-0 data-[swipe=cancel]:transition-[transform_200ms_ease-out] data-[swipe=end]:animate-swipeOut"
                onOpenChange={(open: boolean) => {
                  if (!open) handleRemoveToast(key);
                }}
                key={key}
                asChild
                forceMount
                duration={notification.duration}
              >
                <motion.li
                  className={twMerge(
                    notificationVariants({ variant: notification.variant })
                  )}
                  initial={{
                    y: 0,
                    opacity: 0,
                  }}
                  animate={{
                    y: 0,
                    opacity: 1,
                    transition: { duration: 0.15 },
                  }}
                  exit={{
                    opacity: 0,
                    transition: { duration: 0.15 },
                  }}
                  layout
                >
                  <div className="flex gap-6 md:items-center">
                    {/* Mobile */}
                    <div aria-hidden className="block md:hidden">
                      <IconWrapper
                        variant={notification.variant}
                        size="sm"
                        className="!mt-0"
                      >
                        {notification.variant === "error" && (
                          <IconAlertCircle />
                        )}
                        {notification.variant === "success" && (
                          <IconCheckmarkCircle />
                        )}
                        {notification.variant === "default" && <IconHeart />}
                      </IconWrapper>
                    </div>
                    {/* Desktop */}
                    <div aria-hidden className="hidden md:block">
                      <IconWrapper
                        variant={notification.variant}
                        size="base"
                        className="!mt-0"
                      >
                        {notification.variant === "error" && (
                          <IconAlertCircle />
                        )}
                        {notification.variant === "success" && (
                          <IconCheckmarkCircle />
                        )}
                        {notification.variant === "default" && <IconHeart />}
                      </IconWrapper>
                    </div>
                    <div className="flex flex-col justify-center">
                      <ToastPrimitive.Title className="text-base font-medium  pr-8 leading-6">
                        {notification.title}
                      </ToastPrimitive.Title>

                      {notification.description && (
                        <ToastPrimitive.Description>
                          <DescriptionText variant="secondary">
                            {notification.description}
                          </DescriptionText>
                        </ToastPrimitive.Description>
                      )}
                    </div>
                  </div>
                  <ToastPrimitive.Close
                    className="absolute right-2 top-2 rounded-md p-1"
                    asChild
                  >
                    <Button
                      colorVariant="muted"
                      size="sm"
                      justIcon
                      aria-label="Close"
                    >
                      <IconX className="h-5 w-5" aria-hidden />
                    </Button>
                  </ToastPrimitive.Close>
                </motion.li>
              </ToastPrimitive.Root>
            );
          })}
        </AnimatePresence>

        <ToastPrimitive.Viewport className="w-full sm:w-[26.25rem] fixed h-fit max-h-[calc(100vh-50px)] overflow-auto pt-4 bottom-0 right-0 pb-3 px-3 md:pr-3 z-top space-y-2 md:space-y-4" />
      </ToastPrimitive.Provider>
    </NotificationContext.Provider>
  );
};

/* -----------------------------------------------------------------------------------------------*/

function useNotification() {
  const context = React.useContext(NotificationContext);
  if (context) return context;
  throw new Error("useNotification must be used within Notifications");
}

/* -----------------------------------------------------------------------------------------------*/

export { Notifications, useNotification };
