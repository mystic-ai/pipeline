interface BadgeGroupProps {
  text: string;
  badgeText?: string;
  badgeLeading?: boolean;
  badgeTrailing?: boolean;
}

export function BadgeGroup({
  text,
  badgeText,
  badgeLeading = false,
  badgeTrailing = false,
}: BadgeGroupProps): JSX.Element {
  const paddingClasses = badgeText
    ? badgeLeading
      ? `pl-1 pr-4`
      : "pr-1 pl-4"
    : "px-3";

  return (
    <div
      className={`bg-primary-100 flex gap-3 py-1 rounded-full items-center ${paddingClasses}`}
    >
      {badgeText && badgeLeading && (
        <div className="flex h-full items-center px-[.625rem] bg-primary-400 rounded-full">
          <span className="text-sm font-medium text-white">{badgeText}</span>
        </div>
      )}

      <span className="text-sm font-medium text-primary-700">{text}</span>

      {badgeText && badgeTrailing && (
        <div className="px-[.625rem] bg-primary-400 rounded-full">
          <span className="text-sm font-medium text-primary-700">
            {badgeText}
          </span>
        </div>
      )}
    </div>
  );
}
