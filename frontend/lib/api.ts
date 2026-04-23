const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const ERR_UNAUTHORIZED = "UNAUTHORIZED";

export const CLASS_NAMES = [
  "Pepper__bell___Bacterial_spot",
  "Pepper__bell___healthy",
  "Potato___Early_blight",
  "Potato___Late_blight",
  "Potato___healthy",
  "Tomato_Early_blight",
  "Tomato_Late_blight",
  "Tomato_healthy",
];

export function formatClassName(name: string): string {
  return name
    .replace(/__+/g, " - ")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

//从 localStorage 里取 token
function getToken(): string | null {
  if (typeof window === "undefined") return null; //若是在浏览器里，就没有localStorage
  return localStorage.getItem("token");
}

//自动拼 Authorization 请求头
function authHeaders(): Record<string, string> {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

//从 localStorage 里取用户名
export function getUsername(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("username");
}

//判断是否已登录
export function isLoggedIn(): boolean {
  return !!getToken(); //!!作用是把值强制转成布尔值
}

//退出登录
export function logout() {
  localStorage.removeItem("token");
  localStorage.removeItem("username");
}

//登录接口
export async function login(username: string, password: string) {
  const res = await fetch(`${API_BASE}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) throw new Error("Invalid credentials");
  const data = await res.json();
  localStorage.setItem("token", data.token);
  localStorage.setItem("username", data.username);
  return data;
}

export interface PredictionItem {
  label: string;
  confidence: number;
}

export interface PredictResult {
  record_id: number;
  predicted_class: string;
  confidence: number;
  top_k: PredictionItem[];
  gradcam_heatmap: string;
  overlay_image: string;
}

export async function predict(file: File): Promise<PredictResult> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: authHeaders(),
    body: form,
  });
  if (res.status === 401) throw new Error(ERR_UNAUTHORIZED);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Prediction failed");
  }
  return res.json();
}

export interface HistoryItem {
  record_id: number;
  original_filename: string;
  predicted_class: string;
  confidence: number;
  uploaded_at: string;
  user_feedback_correct: boolean | null;
  user_feedback_label: string | null;
}

export async function getHistory(): Promise<HistoryItem[]> {
  //不写method的话默认是GET
  const res = await fetch(`${API_BASE}/predictions/history`, {
    headers: authHeaders(),
  });
  if (res.status === 401) throw new Error(ERR_UNAUTHORIZED);
  if (!res.ok) throw new Error("Failed to fetch history");
  return res.json();
}

//浏览器在加载 <img> 时会自己发请求的话，没有 Authorization，所以需要手动fetch
export async function fetchImageBlob(recordId: number): Promise<string> {
  const res = await fetch(`${API_BASE}/predictions/${recordId}/image`, {
    headers: authHeaders(),
  });
  if (res.status === 401) throw new Error(ERR_UNAUTHORIZED);
  if (!res.ok) throw new Error("Image not found");
  const blob = await res.blob(); //拿图片二进制数据
  return URL.createObjectURL(blob); //转换成浏览器可用的临时 URL
}

export async function submitFeedback(
  recordId: number,
  isCorrect: boolean,
  correctLabel?: string,
) {
  const res = await fetch(`${API_BASE}/predictions/${recordId}/feedback`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({
      is_correct: isCorrect,
      correct_label: correctLabel ?? null,
    }),
  });
  if (!res.ok) throw new Error("Failed to submit feedback");
  return res.json();
}
