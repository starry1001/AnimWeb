# Human Evaluation Guideline

## Task Overview

In each comparison, you will be given a **reference video** showing an animated webpage and recordings of **two candidate webpages** (Example 1 and Example 2) that attempt to replicate it. Your task is to judge, **separately for static appearance and animation**, which candidate is closer to the reference.

Watch all three videos in full before making a judgment. You may pause or replay them as needed. The candidates are presented in random order, and their model identities are hidden.

## Part I: Static Appearance

Focus on the visually settled states throughout the video rather than transitional frames.

**Step 1: State Check**

- Identify the distinct states in the reference, such as a loading screen, the page before and after a background transition, and the final page.
- Check whether each state is reproduced. A candidate that omits a state preserved by the other should be considered less similar.

**Step 2: Content Check**

- Within each state, check whether the reference text, images, icons, and embedded videos are present.
- A candidate with substantially more missing or extra content, especially prominent elements such as headlines and main images, should be considered less similar.

**Step 3: Layout Check**

- If both candidates are comparable in content, compare the arrangement of their elements.
- Check whether element positions, sizes, spacing, and grouping match the reference.

**Step 4: Style Check**

- If both candidates are comparable in content and layout, compare their styling, including fonts, colors, and text sizes.

**Overall Judgment**

Based on the criteria above, in the order of priority **State > Content > Layout > Style**, judge which candidate (Example 1 or Example 2) more closely matches the static appearance of the reference webpage.

## Part II: Animation

Focus on how the page changes over time, judging animation separately from static appearance.

**Step 1: Motion Presence**

- Identify which components are animated in the reference and check whether they are also animated in each candidate.
- A candidate that omits more of the reference animations or leaves animated components static should be considered less similar.

**Step 2: Timing and Order**

- If both candidates reproduce comparable sets of animations, compare when these animations begin and end, and whether they occur in the same order or overlap as in the reference.
- For example, components revealed one after another should not appear all at once or in a different sequence.

**Step 3: Animation Type**

- If both candidates are comparable in timing and order, check whether their effects match those in the reference.
- For example, a background revealed by a curtain-like sweep should not simply fade in, and an icon that rotates should not merely scale up.

**Step 4: Motion Quality**

- If both candidates use comparable effects, compare how closely the animations resemble the reference as they unfold.
- Consider the components’ paths and directions, their appearance and size during motion, and their speed and changes in speed.

**Overall Judgment**

Based on the criteria above, in the order of priority **Motion Presence > Timing and Order > Animation Type > Motion Quality**, judge which candidate (Example 1 or Example 2) more closely matches the animation of the reference webpage.

## Judgment Options

Make **two separate judgments**, one for static appearance and one for animation:

1. **Example 1 better:** Example 1 is closer to the reference.
2. **Example 2 better:** Example 2 is closer to the reference.
3. **Tie:** Both examples are similarly accurate or equally distant from the reference.

