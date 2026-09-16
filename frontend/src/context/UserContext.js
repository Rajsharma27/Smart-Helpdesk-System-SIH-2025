"use client";

import { createContext, useContext, useEffect, useState } from "react";
import { attemptFetchMe } from "@/services/apiTicket";

const UserContext = createContext(null);

export const UserProvider = ({ children }) => {
  const [isLoading, setIsLoading] = useState(true); // Start with true to check auth on mount
  const [currentUser, setCurrentUser] = useState(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  // Initialize auth state from localStorage on mount
  useEffect(() => {
    if (typeof window !== "undefined") {
      const savedUser = localStorage.getItem("user");
      const loggedInStatus = localStorage.getItem("isLoggedIn") === "true";

      if (savedUser && loggedInStatus) {
        setCurrentUser(JSON.parse(savedUser));
        setIsLoggedIn(true);
      }
      setIsLoading(false);
    }
  }, []);

  // Fetch fresh user data when logged in
  useEffect(() => {
    async function fetchUser() {
      try {
        const result = await attemptFetchMe();
        if (result.status === "success") {
          console.log("User fetched successfully");
          console.log(result.data);
          setCurrentUser(result.data);
          localStorage.setItem("user", JSON.stringify(result.data));
        }
      } catch (error) {
        console.log(error);
        // If fetch fails, clear auth state
        setIsLoggedIn(false);
        setCurrentUser(null);
        localStorage.removeItem("user");
        localStorage.removeItem("isLoggedIn");
      }
    }

    if (isLoggedIn && !isLoading) {
      fetchUser();
    }
  }, [isLoggedIn, isLoading]);

  // Update isLoggedIn when currentUser changes
  useEffect(() => {
    if (currentUser) {
      setIsLoggedIn(true);
      localStorage.setItem("isLoggedIn", "true");
    } else {
      setIsLoggedIn(false);
      localStorage.setItem("isLoggedIn", "false");
    }
  }, [currentUser]);

  return (
    <UserContext.Provider
      value={{
        currentUser,
        setCurrentUser,
        isLoading,
        setIsLoading,
        isLoggedIn,
        setIsLoggedIn, // Also expose this so login can set it
      }}
    >
      {children}
    </UserContext.Provider>
  );
};

export function useUser() {
  const context = useContext(UserContext);
  if (!context) {
    throw new Error("useUser must be used within a UserProvider.");
  }
  return context;
}
