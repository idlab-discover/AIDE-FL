package impl

import "testing"

func TestKaggleDatasetPath(t *testing.T) {
	t.Parallel()

	path, err := kaggleDatasetPath("samybtt-ermds-x")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if path != "samybtt/ermds-x" {
		t.Fatalf("unexpected dataset path %q", path)
	}
}

func TestKaggleDatasetPathRejectsMissingSeparator(t *testing.T) {
	t.Parallel()

	if _, err := kaggleDatasetPath("ermdsx"); err == nil {
		t.Fatal("expected an error for a dataset without owner separator")
	}
}
