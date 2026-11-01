/* Two controls shared by Your block and Try to break the result, drawn in the page's own style so that a choice
   never opens a list from the operating system.

     .segmented            a group of radio pills laid out as a grid of equal cells. A lit thumb glides to the chosen
                           pill with a slight spring. The radios stay native: arrow keys, Tab and screen readers work
                           as usual, and the thumb only follows what the radios say.
     NYC.picker(options)   a menu button with a listbox: the closed control is the same height and width as its
                           neighbours, the list opens just under it at the same width, the chosen option carries a
                           check, and it is driven fully from the keyboard (arrows, Home, End, typing a letter,
                           Enter, Escape). Returns { el, set(value), get() }. */
(function () {
  "use strict";

  var NYC = window.NYC = window.NYC || {};

  /* ---------------------------------------------------------------- the gliding thumb */
  function placeThumb(group, thumb, instant) {
    var chosen = group.querySelector("input:checked");
    var cell = chosen ? chosen.closest(".segment") : null;
    if (!cell) { thumb.style.opacity = "0"; return; }
    if (instant || NYC.reducedMotion()) { thumb.classList.add("is-instant"); }
    thumb.style.opacity = "1";
    thumb.style.width = cell.offsetWidth + "px";
    thumb.style.height = cell.offsetHeight + "px";
    thumb.style.transform = "translate(" + cell.offsetLeft + "px," + cell.offsetTop + "px)";
    if (instant || NYC.reducedMotion()) { void thumb.offsetWidth; thumb.classList.remove("is-instant"); }
  }
  NYC.onMount(".segmented", function (group) {
    // Dash writes its radios without a shared name, so each is a group of one: Tab stops on every pill and the
    // arrow keys do nothing. One name per group restores the native behaviour (one Tab stop, arrows choose).
    var groupName = "grp-" + (group.id || Math.random().toString(36).slice(2));
    group.querySelectorAll("input[type=radio]").forEach(function (input) { if (!input.name) { input.name = groupName; } });
    var thumb = document.createElement("span");
    thumb.className = "seg-thumb"; thumb.setAttribute("aria-hidden", "true");
    group.appendChild(thumb);          // last: React inserts its own children before nodes it knows, so this is left alone
    group.classList.add("has-thumb");
    function settle() { placeThumb(group, thumb, false); }
    // React sets `checked` as a property, which no observer sees; the input events and one frame later cover it.
    group.addEventListener("change", function () { settle(); window.requestAnimationFrame(settle); });
    group.addEventListener("click", function () { window.requestAnimationFrame(settle); });
    if ("ResizeObserver" in window) { new ResizeObserver(function () { placeThumb(group, thumb, true); }).observe(group); }
    placeThumb(group, thumb, true);
    group.__placeThumb = settle;
  });
  NYC.placeThumbs = function (root) {
    (root || document).querySelectorAll(".segmented").forEach(function (group) { if (group.__placeThumb) { group.__placeThumb(); } });
  };

  /* A radio group built in the browser, with the same markup Dash writes for dcc.RadioItems. */
  var groupCount = 0;
  NYC.segmented = function (spec) {
    var name = "seg-" + (++groupCount);
    var field = document.createElement("div");
    field.className = "segmented-field";
    var label = document.createElement("span");
    label.className = "field-label"; label.id = spec.id + "-label"; label.textContent = spec.label;
    var group = document.createElement("div");
    group.className = "segmented"; group.id = spec.id;
    group.setAttribute("role", "radiogroup"); group.setAttribute("aria-labelledby", label.id);
    if (spec.cols) { group.style.setProperty("--cols", spec.cols); }
    var inputs = spec.options.map(function (option) {
      var cell = document.createElement("label"), input = document.createElement("input");
      cell.className = "segment";
      input.type = "radio"; input.name = name; input.value = option[0];
      input.checked = option[0] === spec.value;
      cell.appendChild(input); cell.appendChild(document.createTextNode(option[1]));
      group.appendChild(cell);
      input.addEventListener("change", function () { if (input.checked && spec.onChange) { spec.onChange(option[0]); } });
      return input;
    });
    field.appendChild(label); field.appendChild(group);
    return {
      el: field,
      set: function (value) {
        inputs.forEach(function (input) { input.checked = input.value === value; });
        if (group.__placeThumb) { group.__placeThumb(); }
      }
    };
  };

  /* ---------------------------------------------------------------- the picker */
  var open = null;                                                     // at most one picker is open
  document.addEventListener("pointerdown", function (event) {
    if (open && !open.root.contains(event.target)) { open.close(false); }
  });

  NYC.picker = function (spec) {
    var value = spec.value, active = -1, typed = "", typedAt = 0;
    var root = document.createElement("div");
    root.className = "picker";
    var label = document.createElement("span");
    label.className = "field-label"; label.id = spec.id + "-label"; label.textContent = spec.label;
    var trigger = document.createElement("button");
    trigger.type = "button"; trigger.className = "picker-trigger"; trigger.id = spec.id;
    trigger.setAttribute("aria-haspopup", "listbox"); trigger.setAttribute("aria-expanded", "false");
    trigger.setAttribute("aria-controls", spec.id + "-menu");
    trigger.setAttribute("aria-labelledby", label.id + " " + spec.id);
    var shown = document.createElement("span"); shown.className = "picker-value";
    var chevron = document.createElement("span"); chevron.className = "picker-chevron"; chevron.setAttribute("aria-hidden", "true");
    trigger.appendChild(shown); trigger.appendChild(chevron);
    var menu = document.createElement("ul");
    menu.className = "picker-menu"; menu.id = spec.id + "-menu"; menu.tabIndex = -1;
    menu.setAttribute("role", "listbox"); menu.setAttribute("aria-labelledby", label.id);
    var items = spec.options.map(function (option, index) {
      var item = document.createElement("li");
      item.className = "picker-option"; item.id = spec.id + "-opt-" + index;
      item.setAttribute("role", "option");
      item.style.setProperty("--i", index);
      var check = document.createElement("span"); check.className = "picker-check"; check.setAttribute("aria-hidden", "true");
      item.appendChild(check); item.appendChild(document.createTextNode(option[1]));
      item.addEventListener("pointermove", function () { if (active !== index) { setActive(index); } });
      item.addEventListener("click", function () { choose(index); });
      menu.appendChild(item);
      return item;
    });
    root.appendChild(label); root.appendChild(trigger); root.appendChild(menu);

    function indexOf(v) { for (var i = 0; i < spec.options.length; i++) { if (spec.options[i][0] === v) { return i; } } return 0; }
    function paint() {
      var chosen = indexOf(value);
      shown.textContent = spec.options[chosen][1];
      items.forEach(function (item, i) { item.setAttribute("aria-selected", i === chosen ? "true" : "false"); });
    }
    function setActive(index) {
      active = index;
      items.forEach(function (item, i) { item.classList.toggle("is-active", i === index); });
      if (index >= 0) { menu.setAttribute("aria-activedescendant", items[index].id); items[index].scrollIntoView({ block: "nearest" }); }
    }
    function openMenu() {
      if (open && open !== api) { open.close(false); }
      open = api;
      root.classList.add("is-open");
      trigger.setAttribute("aria-expanded", "true");
      // Open upward when there is no room below.
      var box = trigger.getBoundingClientRect(), need = Math.min(items.length * 44 + 16, 320);
      root.classList.toggle("opens-up", window.innerHeight - box.bottom < need && box.top > need);
      setActive(indexOf(value));
      menu.focus({ preventScroll: true });
    }
    function close(refocus) {
      root.classList.remove("is-open");
      trigger.setAttribute("aria-expanded", "false");
      menu.removeAttribute("aria-activedescendant");
      if (open === api) { open = null; }
      if (refocus) { trigger.focus({ preventScroll: true }); }
    }
    function choose(index) {
      var next = spec.options[index][0], changed = next !== value;
      value = next; paint(); close(true);
      if (changed) {
        trigger.classList.remove("is-chosen"); void trigger.offsetWidth; trigger.classList.add("is-chosen");
        if (spec.onChange) { spec.onChange(value); }
      }
    }

    trigger.addEventListener("click", function () { if (root.classList.contains("is-open")) { close(true); } else { openMenu(); } });
    trigger.addEventListener("keydown", function (event) {
      if (event.key === "ArrowDown" || event.key === "ArrowUp" || event.key === "Enter" || event.key === " ") { event.preventDefault(); openMenu(); }
    });
    menu.addEventListener("keydown", function (event) {
      var last = items.length - 1;
      if (event.key === "ArrowDown") { event.preventDefault(); setActive(Math.min(last, active + 1)); }
      else if (event.key === "ArrowUp") { event.preventDefault(); setActive(Math.max(0, active - 1)); }
      else if (event.key === "Home") { event.preventDefault(); setActive(0); }
      else if (event.key === "End") { event.preventDefault(); setActive(last); }
      else if (event.key === "Enter" || event.key === " ") { event.preventDefault(); if (active >= 0) { choose(active); } }
      else if (event.key === "Escape") { event.preventDefault(); close(true); }
      else if (event.key === "Tab") { close(false); }
      else if (event.key.length === 1 && /\S/.test(event.key)) {
        var now = performance.now();
        typed = (now - typedAt < 700 ? typed : "") + event.key.toLowerCase(); typedAt = now;
        for (var i = 0; i < spec.options.length; i++) {
          if (spec.options[i][1].toLowerCase().indexOf(typed) === 0) { setActive(i); break; }
        }
      }
    });

    var api = { el: root, root: root, close: close, get: function () { return value; },
                set: function (v) { value = v; paint(); } };
    paint();
    return api;
  };
})();
