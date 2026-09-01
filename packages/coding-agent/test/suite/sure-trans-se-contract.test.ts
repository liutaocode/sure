import { describe, expect, it } from "vitest";
import { findUnit } from "../../../../sure/skills/sure_trans/hooks/state-machine.ts";

describe("sure_trans speech enhancement task contract", () => {
	it("persists only the canonical se task value", () => {
		const values = findUnit("load_trans_input")?.allowedValues?.task_type;

		expect(values).toContain("se");
		expect(values).not.toContain("speech-enhancement");
		expect(values).not.toContain("speech_enhancement");
	});
});
