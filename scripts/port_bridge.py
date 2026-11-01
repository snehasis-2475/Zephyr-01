import http.server
import socketserver
import threading
import sys

TARGET = "http://localhost:8443"

class RedirectHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(302)
        self.send_header('Location', f"{TARGET}{self.path}")
        self.end_headers()

    def do_HEAD(self):
        self.send_response(302)
        self.send_header('Location', f"{TARGET}{self.path}")
        self.end_headers()

    def log_message(self, format, *args):
        pass

def serve_port(port):
    class DualStackServer(socketserver.TCPServer):
        allow_reuse_address = True

    try:
        with DualStackServer(("", port), RedirectHandler) as httpd:
            print(f"[PortBridge] Listening on port {port} -> forwarding to {TARGET}")
            httpd.serve_forever()
    except Exception as e:
        print(f"[PortBridge] Port {port} error: {e}")

if __name__ == "__main__":
    ports = [5173, 3000]
    threads = []
    for p in ports:
        t = threading.Thread(target=serve_port, args=(p,), daemon=True)
        t.start()
        threads.append(t)
    print(f"[PortBridge] Active on ports {ports}. Redirecting to {TARGET}.")
    for t in threads:
        t.join()
