// Headless-ish page driver for macOS (WKWebView): loads a URL, runs a small script of steps, prints
// console errors / JS results and writes PNG screenshots. No Chrome needed.
//   swiftc -O ui/tools/wkshot.swift -o /tmp/wkshot
//   /tmp/wkshot 'http://127.0.0.1:5173/?mock=1' steps.txt
// steps.txt lines:  size W H | wait SECONDS | js EXPRESSION | snap /path.png | click CSS-SELECTOR | key K
import Cocoa
import WebKit

final class Driver: NSObject, WKScriptMessageHandler, WKNavigationDelegate {
    var web: WKWebView!
    var window: NSWindow!
    var steps: [String] = []
    var idx = 0
    var errors: [String] = []

    func userContentController(_ c: WKUserContentController, didReceive m: WKScriptMessage) {
        errors.append("\(m.body)")
        print("[console] \(m.body)")
    }

    func start(url: String, steps: [String]) {
        self.steps = steps
        let conf = WKWebViewConfiguration()
        let ucc = WKUserContentController()
        let js = """
        (function(){
          var post = function(kind, args){ try { window.webkit.messageHandlers.log.postMessage(kind + ': ' + Array.prototype.map.call(args, function(a){ try { return (a && a.stack) ? a.stack : (typeof a === 'object' ? JSON.stringify(a) : String(a)); } catch(e){ return String(a); } }).join(' ')); } catch(e){} };
          ['error','warn'].forEach(function(k){ var o = console[k]; console[k] = function(){ post(k, arguments); o.apply(console, arguments); }; });
          window.addEventListener('error', function(e){ post('window.onerror', [e.message + ' @ ' + e.filename + ':' + e.lineno]); });
          window.addEventListener('unhandledrejection', function(e){ post('unhandledrejection', [e.reason]); });
        })();
        """
        ucc.addUserScript(WKUserScript(source: js, injectionTime: .atDocumentStart, forMainFrameOnly: false))
        ucc.add(self, name: "log")
        conf.userContentController = ucc
        conf.websiteDataStore = WKWebsiteDataStore.nonPersistent()
        web = WKWebView(frame: NSRect(x: 0, y: 0, width: 1440, height: 900), configuration: conf)
        web.navigationDelegate = self
        window = NSWindow(contentRect: NSRect(x: -2200, y: 100, width: 1440, height: 900), styleMask: [.titled], backing: .buffered, defer: false)
        window.contentView = web
        window.orderFrontRegardless()
        web.load(URLRequest(url: URL(string: url)!))
    }

    func webView(_ w: WKWebView, didFinish n: WKNavigation!) { print("[nav] loaded"); next() }
    func webView(_ w: WKWebView, didFail n: WKNavigation!, withError e: Error) { print("[nav] FAIL \(e)"); exit(2) }
    func webView(_ w: WKWebView, didFailProvisionalNavigation n: WKNavigation!, withError e: Error) { print("[nav] FAIL \(e)"); exit(2) }

    func next() {
        if idx >= steps.count {
            print("[done] console errors/warnings: \(errors.count)")
            exit(errors.isEmpty ? 0 : 1)
        }
        let line = steps[idx]; idx += 1
        let parts = line.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: true).map(String.init)
        guard let cmd = parts.first else { return next() }
        let arg = parts.count > 1 ? parts[1] : ""
        switch cmd {
        case "size":
            let wh = arg.split(separator: " ").compactMap { Double($0) }
            if wh.count == 2 {
                window.setContentSize(NSSize(width: wh[0], height: wh[1]))
                web.frame = NSRect(x: 0, y: 0, width: wh[0], height: wh[1])
            }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { self.next() }
        case "wait":
            DispatchQueue.main.asyncAfter(deadline: .now() + (Double(arg) ?? 1)) { self.next() }
        case "js":
            web.evaluateJavaScript(arg) { r, e in
                if let e = e { print("[js] ERROR \(e)"); self.errors.append("js: \(e)") } else { print("[js] \(r ?? "nil")") }
                self.next()
            }
        case "click":
            let s = "(function(){var e=document.querySelector(\(jsString(arg)));if(!e)return 'NOT FOUND';e.click();return 'clicked';})()"
            web.evaluateJavaScript(s) { r, _ in print("[click \(arg)] \(r ?? "nil")"); DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { self.next() } }
        case "key":
            let s = "window.dispatchEvent(new KeyboardEvent('keydown',{key:\(jsString(arg)),bubbles:true}))"
            web.evaluateJavaScript(s) { _, _ in DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { self.next() } }
        case "snap":
            let cfg = WKSnapshotConfiguration()
            web.takeSnapshot(with: cfg) { img, e in
                if let img = img, let tiff = img.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff), let png = rep.representation(using: .png, properties: [:]) {
                    try? png.write(to: URL(fileURLWithPath: arg)); print("[snap] \(arg)")
                } else { print("[snap] FAILED \(String(describing: e))") }
                self.next()
            }
        default:
            print("[?] unknown step \(line)"); next()
        }
    }
}

func jsString(_ s: String) -> String {
    let data = try! JSONSerialization.data(withJSONObject: [s], options: [])
    let str = String(data: data, encoding: .utf8)!
    return String(str.dropFirst().dropLast())
}

let args = CommandLine.arguments
guard args.count >= 3 else { print("usage: wkshot URL steps.txt"); exit(64) }
let steps = (try! String(contentsOfFile: args[2])).split(separator: "\n").map(String.init).filter { !$0.hasPrefix("#") && !$0.isEmpty }
let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let d = Driver()
d.start(url: args[1], steps: steps)
DispatchQueue.main.asyncAfter(deadline: .now() + 120) { print("[timeout]"); exit(3) }
app.run()
