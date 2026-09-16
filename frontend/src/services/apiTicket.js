import { getToken } from "@/lib/utils";

const DEFAULT_API = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:4000/api";

const API = DEFAULT_API;

export async function attemptRegister(userData) {
  const result = await fetch(`${API}/users`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(userData),
    credentials: "include",
  });
  const data = await result.json();

  if (typeof window !== "undefined" && data.status === "success") {
    window.localStorage.setItem("isLoggedIn", true);
    window.localStorage.setItem("token", data.token);
  }

  return data;
}

export async function attemptLogin({ email, password }) {
  const result = await fetch(`${API}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
    credentials: "include",
  });
  const data = await result.json();

  if (typeof window !== "undefined" && data.status === "success") {
    // Perform localStorage operations only in the browser environment
    window.localStorage.setItem("isLoggedIn", true);
    window.localStorage.setItem("token", data.token);
  }

  return data;
}

export async function attemptFetchMe() {
  if (typeof window === "undefined") return;
  const token = window.localStorage.getItem("token");
  if (!token) {
    console.log("No token found, cannot fetch user details without token");
    return;
  }
  const result = await fetch(`${API}/users/me`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    credentials: "include",
  });

  return result.json();
}

export async function attemptLogout() {
  if (typeof window !== "undefined") {
    localStorage.removeItem("isLoggedIn");
    localStorage.removeItem("token");
    localStorage.removeItem("user");
  }
}

export async function fetchAllMyTickets() {
  const result = await fetch(`${API}/tickets/my-tickets`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${getToken()}`,
    },
    credentials: "include",
  });
  console.log("Fetching my tickets from API:", result);
  return result.json();
}

export async function createTicket(ticketData) {
  const result = await fetch(`${API}/tickets`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${getToken()}`,
    },
    body: JSON.stringify(ticketData),
    credentials: "include",
  });
  return result.json();
}

export async function updateTicket(ticketId, updateData) {
  const result = await fetch(`${API}/tickets/${ticketId}`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${getToken()}`,
    },
    body: JSON.stringify(updateData),
    credentials: "include",
  });
  return result.json();
}
