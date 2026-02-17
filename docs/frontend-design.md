
  ---
  Technology Stack

  Layer: Framework
  Choice: Astro 5.x (static output)
  Why it matters: Zero JS shipped by default; pages are pure HTML/CSS until you opt in
  ────────────────────────────────────────
  Layer: Styling
  Choice: Tailwind CSS v4 via @tailwindcss/vite
  Why it matters: Not the older @astrojs/tailwind integration. Tokens live in src/styles/global.css
    inside @theme {}
  ────────────────────────────────────────
  Layer: Type safety
  Choice: TypeScript (strict)
  Why it matters: Every component has a typed Props interface
  ────────────────────────────────────────
  Layer: Blog/content
  Choice: Astro Content Collections with glob() loader
  Why it matters: Markdown/MDX posts in src/content/blog/, schema in src/content.config.ts
  ────────────────────────────────────────
  Layer: SEO
  Choice: @astrojs/sitemap, custom SEOHead.astro, src/utils/seo.ts
  Why it matters: JSON-LD schemas (Organization, BlogPosting, FAQ, Breadcrumb), OG tags, RSS
  ────────────────────────────────────────
  Layer: Utility
  Choice: clsx + tailwind-merge wrapped as cn() in src/utils/merge.ts
  Why it matters: Every component that accepts a class prop uses this for conflict-free merging
  ────────────────────────────────────────
  Layer: Deps that are NOT used
  Choice: No React, no anime.js, no astro-navbar
  Why it matters: Despite what DESIGN.md says (it describes the Zen Browser site this was adapted
    from). All interactivity is vanilla <script> or CSS-only

  ---
  Color System

  All colors are CSS custom properties that flip between light and dark mode. Defined in global.css:

  :root                      [data-theme="dark"]
  --zen-paper: #f2f0e3       --zen-paper: #1f1f1f
  --zen-dark:  #2e2e2e       --zen-dark:  #d1cfc0
  --zen-muted: rgba(0,0,0,0.04)   rgba(255,255,255,0.04)
  --zen-subtle: rgba(0,0,0,0.06)  rgba(255,255,255,0.07)

  These are mapped to Tailwind via @theme {}:
  - paper / dark -- the two "ink on paper" roles that invert in dark mode
  - coral (#F76F53) -- primary accent for CTAs, links, highlights
  - zen-blue (#6287f5) -- secondary accent
  - zen-green (#63f78b) -- tertiary accent / success

  The overall feel is warm, not clinical. The light mode background is a parchment off-white, not
  pure white. Dark mode is a warm charcoal, not pure black.

  Opacity pattern: Instead of defining dozens of gray shades, the site uses text-dark/50,
  text-dark/70, border-dark/[0.06], etc. -- Tailwind's opacity modifier on the base dark color. This
   is used everywhere for secondary text, borders, and muted surfaces.

  ---
  Typography

  Two fonts, each with a clear role:

  1. Bricolage Grotesque (body, everything) -- variable sans-serif loaded via @fontsource at weights
   400-700. Applied to body with font-weight: 500 as default, font-variation-settings: 'width' 100.
  2. Junicode (display headings only) -- variable serif, self-hosted woff2 in public/fonts/. Two
  variants: Roman and Italic. Applied via font-junicode class. The italic variant is used on accent
  words inside <h1> with the italic class, but importantly font-style: normal because the italic is
  baked into the separate font file. Swash features (swsh 1 on Roman, swsh 0 on Italic).

  Heading sizes scale responsively: text-3xl sm:text-4xl lg:text-5xl is the most common pattern for
  section headings. The hero goes up to text-7xl on desktop.

  ---
  Layout Architecture

  Layout chain

  Page
    -> PageLayout  (Navbar + skip-to-content + <main> + Footer)
      -> BaseLayout (html shell, SEOHead, global.css, inline scripts)

  Blog posts use BlogLayout -> BaseLayout directly, skipping the Navbar/Footer.

  Container pattern

  No Tailwind container class. Instead, every section uses:
  mx-auto max-w-6xl px-4 sm:px-6 lg:px-8
  The Navbar uses max-w-7xl for slightly wider reach.

  Section pattern

  Every landing page section follows this structure:
  <section id="section-name" class="section-divider py-12 lg:py-24">
    <div class="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
      <!-- Header block with data-animate -->
      <div class="mb-12 text-center lg:mb-16" data-animate>
        <h2 class="text-3xl font-semibold leading-[0.95] tracking-tight sm:text-4xl
  lg:text-5xl">...</h2>
        <p class="mx-auto mt-5 max-w-2xl text-base text-dark/60 sm:text-lg">...</p>
      </div>
      <!-- Content -->
    </div>
  </section>

  Key conventions:
  - section-divider adds a subtle horizontal gradient line at the top (::before pseudo-element, 1px,
   fades from transparent to --zen-dark and back, 8% opacity)
  - Section padding: py-12 lg:py-24 to py-16 lg:py-36 depending on importance
  - Section headings: centered, tracking-tight, leading-[0.95]
  - Subtext under headings: text-dark/60 (60% opacity of the base text color)
  - Some sections have a colored pill/badge above the heading (e.g. Pricing uses rounded-full
  bg-coral/10 text-coral with text-[11px] uppercase tracking-[0.2em])

  ---
  Dark Mode

  Detection: A blocking inline <script> in BaseLayout runs before first paint:
  1. Check localStorage.getItem('theme')
  2. Fall back to prefers-color-scheme
  3. Default to 'light'
  4. Set data-theme attribute on <html>

  Toggle: Second inline script attaches click handlers to #theme-toggle (desktop) and
  .theme-toggle-mobile buttons. Swaps the attribute, toggles a dark class, persists to localStorage,
   and updates sun/moon icon visibility.

  CSS strategy: All theme-aware colors use the CSS custom properties. Borders and backgrounds use
  opacity modifiers. Some components have manual dark overrides like [data-theme="dark"] .shadow-lg
  { --tw-shadow-color: rgba(0,0,0,0.4) }.

  ---
  Animation System

  Entirely CSS-based, triggered by a small IntersectionObserver in BaseLayout:

  Scroll-triggered animations

  Elements get a data-animate attribute. The observer adds is-visible class when they enter viewport
   (threshold 5%, rootMargin -40px bottom), then unobserves. Three animation types:

  ┌────────────────────────┬───────────┬───────────────────────────────────────────────┐
  │       Attribute        │ Keyframes │                    Effect                     │
  ├────────────────────────┼───────────┼───────────────────────────────────────────────┤
  │ data-animate (default) │ zenReveal │ blur(4px) + translateY(24px) + opacity 0 -> 1 │
  ├────────────────────────┼───────────┼───────────────────────────────────────────────┤
  │ data-animate="fade"    │ zenFade   │ blur(4px) + opacity 0 -> 1                    │
  ├────────────────────────┼───────────┼───────────────────────────────────────────────┤
  │ data-animate="scale"   │ zenScale  │ blur(4px) + scale(0.95) + opacity 0 -> 1      │
  └────────────────────────┴───────────┴───────────────────────────────────────────────┘

  Stagger via data-delay="1" through data-delay="6" (increments of 0.15s). All use
  cubic-bezier(0.25, 0.1, 0.25, 1) easing, 0.5s duration.

  Hero entrance

  Uses .hero-child class with a CSS custom property --hero-delay. Plays on page load (not
  scroll-triggered). Same blur+translate+opacity pattern with heroEntrance keyframes, 0.6s duration,
   delay = var(--hero-delay) * 0.15s + 0.1s.

  Reduced motion

  @media (prefers-reduced-motion: reduce) disables all animations, resets opacity/transform/filter
  to final values.

  ---
  Component Patterns

  Button (ui/Button.astro)

  Polymorphic -- renders <a> if href is given, <button> otherwise. Three variants:
  - Primary (isPrimary): bg-dark text-paper shadow-lg -- filled, dark background
  - Bordered (isBordered): border-2 border-dark/80 that fills on hover
  - Default: bg-subtle (links) or unstyled (buttons)

  All get hover:scale-[1.02] active:scale-[0.98] micro-interaction, 200ms transition, rounded-xl for
   links / rounded-lg for buttons.

  Card (ui/Card.astro)

  Polymorphic element tag (div/article/section). Base: rounded-xl bg-subtle/80 border
  border-dark/[0.08]. Hover: lifts up 2px (-translate-y-0.5), increases border opacity, adds shadow.
   300ms transition.

  Navbar

  3-column grid at desktop (grid-cols-[auto_1fr_auto]): logo left, centered nav, right-side actions.
   Dropdowns are CSS-hover based with a JS click fallback. Desktop nav hidden below lg, hamburger
  hidden above lg.

  Mobile Menu

  CSS-only toggle using a hidden checkbox (#mobile-menu-toggle) + Tailwind's peer modifier. Overlay
  fades in, panel slides from right (translate-x-full -> translate-x-0), 300ms. Nested <details> for
   collapsible groups.

  Footer

  Inverted colors -- bg-dark text-paper. Decorative concentric circle rings (paper-colored borders
  at 18% opacity) positioned bottom-right, hidden on mobile. Link columns in a 2-col / 3-col grid.

  ---
  Interactive Patterns (minimal JS)

  The site avoids framework JS. All interactivity is <script> tags (not is:inline for scoped
  scripts, is:inline for global ones):

  1. Theme toggle -- inline script in BaseLayout
  2. Scroll animations -- inline IntersectionObserver in BaseLayout
  3. Feature tabs -- scoped <script> in Features.astro, toggles data-active attributes and classes
  4. Navbar dropdowns -- scoped <script> in Navbar.astro, click handler + click-outside-to-close
  5. Mobile menu close on hash-link -- scoped <script> in MobileMenu.astro

  All scoped scripts re-initialize on astro:after-swap for view transitions compatibility.

  ---
  Visual Motifs to Replicate

  1. Art backgrounds under app mockups: The hero and feature panels overlay a dark semi-transparent
  UI mockup on top of a painting/artwork image. Pattern: <img> fills the container, bg-dark/20
  overlay, then a bg-dark/95 backdrop-blur-sm card floats on top.
  2. Window chrome: Fake browser/app window headers with three dots (h-3 w-3 rounded-full
  bg-paper/20) and an address bar.
  3. Checkerboard icon grid: The Integrations section uses a CSS grid with alternating shaded cells
  ((row + col) % 2 === 1). Each cell has a single-color SVG logo. Hover scales to 1.08.
  4. Concentric circles: Decorative motif used in the logo SVG and Footer. Concentric rounded-full
  divs with thick borders at low opacity.
  5. Subtle borders everywhere: Not solid lines but very low-opacity: border-dark/[0.04] for navbar,
   border-dark/[0.06] for dividers, border-dark/[0.08] for cards.
  6. Comparison table for pricing: 4-column grid (grid-cols-4), check/X SVGs for boolean features,
  highlighted middle column with coral accents. Mobile collapses to stacked cards that hide false
  features.
  7. FAQ with native <details>: No JS accordion. <details> + <summary> with a chevron that rotates
  via group-open:rotate-180.

  ---
  What to Reuse vs. Adapt for a Docs Site

  Reuse directly (copy from this repo):
  - global.css (color tokens, font faces, animation keyframes, base styles)
  - BaseLayout.astro (theme detection, animation observer)
  - Navbar.astro and MobileMenu.astro (change links, keep structure)
  - Footer.astro (change content)
  - Button.astro, Card.astro (generic UI primitives)
  - cn() utility, seo.ts schemas
  - Content collection config (for docs pages instead of blog posts)

  Adapt:
  - Replace marketing sections (Hero, Pricing, FAQ, etc.) with docs-specific layouts: sidebar
  navigation, prose content area, table of contents
  - The content collection schema -- docs likely need order/category fields instead of pubDate/tags
  - PageLayout -- docs will probably want a persistent sidebar instead of just Navbar + Footer
  wrapping <main>

  Key things to preserve for visual consistency:
  - The paper/dark color tokens and warm palette
  - Bricolage Grotesque for body + Junicode for display headings
  - The section-divider gradient lines
  - The opacity-modifier pattern for secondary text (text-dark/60, text-dark/70)
  - The blur-based scroll animations
  - The rounded-xl / rounded-lg corner radius language
  - The subtle border pattern (border-dark/[0.06])