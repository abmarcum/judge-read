# Judge Read - Frontend Client UI ⚖️💻

The client-side React user interface for the Judge Read legal search and chat assistant. Designed with a clean glassmorphism aesthetic specifically tailored for professional law environments.

---

### 🎨 Advanced Legal UI Layouts
* **Tab Navigation**: Created clean header controls to switch context seamlessly between **Research Chat**, **Case Explorer**, **DB Analytics**, and **Performance Tests** without opening standard popups.
* **Split-Screen Workspace**: Side-by-side reading layout (50% chat / 50% case reader) triggered directly from the case toolbar, allowing attorneys to read case opinions while chatting with the retrieval assistant.
* **Citations Linkify**: Parses legal citations in LLM responses and renders them as interactive pills. Clicking a pill resolves and displays the case text in the viewer.
* **Annotations Selection Popover**: Highlights selected text in the Case Reader, inputs notes into a floating editor, and saves annotations instantly to the database.
* **Research Memo Exporter**: Downloads Markdown-formatted reports of queries and highlights.
* **Benchmarking & Analytics rendering**: Pulls stats on case distribution, runs step-by-step performance tests, and displays data in responsive progress bars.

---

## 🛠️ Tech Stack

* **Core Framework**: React 18
* **Build System**: Vite (Fast HMR)
* **HTTP Client**: Axios (configured with REST request interceptors)
* **Styling**: Vanilla CSS (Variables, Flexbox, CSS Grid)
* **Iconography**: Lucide React

---

## 🚀 Setup & Execution

### 1. Install Node Dependencies
Ensure you have Node.js (version `>= 18`) installed. From the `frontend` folder, install required packages:
```bash
npm install
```

### 2. Start the Vite Dev Server
```bash
npm run dev
```

* **Local Access**: By default, the client starts at `http://localhost:5173`.
* **LAN / Network Access**: To access the UI from other devices on your local network (e.g., testing on a mobile device or tablet), boot Vite with the host flag:
  ```bash
  npm run dev -- --host
  ```

---

## 🌐 Networking & REST API Routing

The React frontend utilizes a dynamic hostname binding strategy to facilitate network deployments:

* **Dynamic Port Association**: All Axios queries are routed to port `8000` using the window's active location hostname:
  ```javascript
  axios.get(`http://${window.location.hostname}:8000/api/...`)
  ```
* **Seamless Local vs. Network Running**:
  * If accessed locally via `http://localhost:5173`, the frontend directs API queries to `http://localhost:8000`.
  * If accessed over the local network via `http://192.168.1.178:5173`, the frontend automatically routes API queries to the backend running at `http://192.168.1.178:8000`.
  * Make sure your backend server is running with the `--host 0.0.0.0` flag if you want to support network connections.
