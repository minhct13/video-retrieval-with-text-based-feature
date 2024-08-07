from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time



options = Options()
options.add_argument("--headless")
options.add_argument("--window-size=1920,1080")
# options.add_argument(f'user-agent={userAgent}')


# options.binary_location=r'C:\Users\USER\Desktop\video-retrieval-with-text-based-feature\chrome-win32\chrome.exe'
service = Service(executable_path="chromedriver_win32/chromedriver")
driver = webdriver.Chrome(service=service, options=options)

driver.get("http://www.google.com")


# Wait for the button to be clickable and then click it
button = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.ID, "btn_vip_show_popup"))
)
button.click()

# Wait for the download link to be generated and extract it
time.sleep(3)  # Adjust sleep time if necessary for the link to be ready
download_link = driver.execute_script("""
var saveAs = saveAs || function(e) {
    "use strict";
    if (typeof e === "undefined" || typeof navigator !== "undefined" && /MSIE [1-9]\./.test(navigator.userAgent)) {
        return
    }
    var t = e.document
      , n = function() {
        return e.URL || e.webkitURL || e
    }
      , r = t.createElementNS("http://www.w3.org/1999/xhtml", "a")
      , o = "download"in r
      , a = function(e) {
        var t = new MouseEvent("click");
        e.dispatchEvent(t)
    }
      , i = /constructor/i.test(e.HTMLElement) || e.safari
      , f = /CriOS\/[\d]+/.test(navigator.userAgent)
      , u = function(t) {
        (e.setImmediate || e.setTimeout)(function() {
            throw t
        }, 0)
    }
      , s = "application/octet-stream"
      , d = 1e3 * 40
      , c = function(e) {
        var t = function() {
            if (typeof e === "string") {
                n().revokeObjectURL(e)
            } else {
                e.remove()
            }
        };
        setTimeout(t, d)
    }
      , l = function(e, t, n) {
        t = [].concat(t);
        var r = t.length;
        while (r--) {
            var o = e["on" + t[r]];
            if (typeof o === "function") {
                try {
                    o.call(e, n || e)
                } catch (a) {
                    u(a)
                }
            }
        }
    }
      , p = function(e) {
        if (/^\s*(?:text\/\S*|application\/xml|\S*\/\S*\+xml)\s*;.*charset\s*=\\s*utf-8/i.test(e.type)) {
            return new Blob([String.fromCharCode(65279), e],{
                type: e.type
            })
        }
        return e
    }
      , v = function(t, u, d) {
        if (!d) {
            t = p(t)
        }
        var v = this, w = t.type, m = w === s, y, h = function() {
            l(v, "writestart progress write writeend".split(" "))
        }, S = function() {
            if ((f || m && i) && e.FileReader) {
                var r = new FileReader;
                r.onloadend = function() {
                    var t = f ? r.result : r.result.replace(/^data:[^;]*;/, "data:attachment/file;");
                    var n = e.open(t, "_blank");
                    if (!n)
                        e.location.href = t;
                    t = undefined;
                    v.readyState = v.DONE;
                    h()
                }
                ;
                r.readAsDataURL(t);
                v.readyState = v.INIT;
                return
            }
            if (!y) {
                y = n().createObjectURL(t)
            }
            console.log("Download link: " + y); // Log the download link
            if (m) {
                e.location.href = y
            } else {
                var o = e.open(y, "_blank");
                if (!o) {
                    e.location.href = y
                }
            }
            v.readyState = v.DONE;
            h();
            c(y)
        };
        v.readyState = v.INIT;
        if (o) {
            y = n().createObjectURL(t);
            setTimeout(function() {
                r.href = y;
                r.download = u;
                a(r);
                h();
                c(y);
                v.readyState = v.DONE
            });
            return
        }
        S()
    }
      , w = v.prototype
      , m = function(e, t, n) {
        return new v(e,t || e.name || "download",n)
    };
    if (typeof navigator !== "undefined" && navigator.msSaveOrOpenBlob) {
        return function(e, t, n) {
            t = t || e.name || "download";
            if (!n) {
                e = p(e)
            }
            return navigator.msSaveOrOpenBlob(e, t)
        }
    }
    w.abort = function() {}
    ;
    w.readyState = w.INIT = 0;
    w.WRITING = 1;
    w.DONE = 2;
    w.error = w.onwritestart = w.onprogress = w.onwrite = w.onabort = w.onerror = w.onwriteend = null;
    return m
}(typeof self !== "undefined" && self || typeof window !== "undefined" && window || this.content);
return saveAs
""")
print("Download link:", download_link)

# Close the driver
driver.quit()

