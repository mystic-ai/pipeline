import { useClickAway } from "ahooks";
import { AnimatePresence, motion } from "framer-motion";
import { ChangeEvent, ReactNode, useEffect, useRef, useState } from "react";
import { InputSearch } from "./InputSearch";
import { DescriptionText } from "../Typography/DescriptionText";

interface Props<T> {
  items: T[];
  renderItem: (item: T) => ReactNode;
  onSelect?: (item: T) => void;
  id: string;
  className?: string;
  placeholder?: string;
  filterFunction: (data: T[], query: string) => T[];
  emptyComponent?: ReactNode;
  defaultValue?: string;
}

export function InputSearchWithResults<T>(props: Props<T>) {
  const {
    items,
    renderItem,
    filterFunction,
    emptyComponent,
    placeholder,
    defaultValue = "",
  } = props;
  // Refs
  const ref = useRef<HTMLDivElement>(null);

  // State
  const [searchText, setSearchText] = useState<string>(defaultValue || "");
  const [searchResults, setSearchResults] = useState<T[]>([]);
  const [showResults, setShowResults] = useState<boolean>(false);

  // Handlers
  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    let value = event.target.value;

    if (value !== "") {
      setSearchText(value);
      const filteredResults = filterFunction(items, value);
      setSearchResults(filteredResults);
      setShowResults(true);
    }
  };

  // Hooks
  useClickAway(() => {
    setShowResults(false);
  }, ref);

  // Effects
  useEffect(() => {
    if (defaultValue !== "") {
      setSearchText(defaultValue);
      const filteredResults = filterFunction(items, defaultValue);
      setSearchResults(filteredResults);
    }
  }, [items]);

  return (
    <div ref={ref} className="relative w-full">
      <InputSearch
        defaultValue={searchText}
        placeholder={placeholder || "Search..."}
        onChange={handleChange}
        onFocus={() => {
          if (searchText !== "") {
            setShowResults(true);
          }
        }}
        onClear={() => {
          setSearchText("");
          setSearchResults([]);
          setShowResults(false);
        }}
      />

      <AnimatePresence>
        {showResults && searchText !== "" ? (
          <motion.ul
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="
              absolute left-0 top-full
              w-full max-h-48 lg:max-h-60 overflow-auto
              mt-2 py-1 rounded shadow-lg duration-75 transition-transform
              bg-white dark:bg-black z-30
              border border-gray-200 dark:border-gray-700"
          >
            {searchResults.length > 0 ? (
              searchResults.map(renderItem)
            ) : emptyComponent ? (
              <DescriptionText className="px-4 py-2">
                {emptyComponent}
              </DescriptionText>
            ) : null}
          </motion.ul>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
