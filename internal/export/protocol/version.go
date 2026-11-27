
package protocol

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

var (
	// ErrInvalidVersion indicates that a protocol version is malformed or
	// outside the range accepted by this implementation.
	ErrInvalidVersion = errors.New("invalid protocol version")

	// ErrIncompatibleVersion indicates that two valid protocol versions
	// cannot communicate directly.
	ErrIncompatibleVersion = errors.New("incompatible protocol version")

	// ErrNoCompatibleVersion indicates that version negotiation could not find
	// a version supported by both peers.
	ErrNoCompatibleVersion = errors.New(
		"no compatible protocol version",
	)
)

// Version identifies one OTGH protocol revision.
//
// OTGH uses semantic-style major/minor/patch numbering:
//
//   Major:
//       Breaking wire/protocol compatibility changes.
//
//   Minor:
//       Backward-compatible protocol capabilities.
//
//   Patch:
//       Compatible corrections and clarifications.
//
// Compatibility policy is intentionally implemented below instead of being
// implied by the numbers themselves.
type Version struct {
	Major uint16 `json:"major"`
	Minor uint16 `json:"minor"`
	Patch uint16 `json:"patch"`
}

const (
	// CurrentMajor is the protocol major version implemented here.
	CurrentMajor uint16 = 1

	// CurrentMinor is the current minor revision.
	CurrentMinor uint16 = 0

	// CurrentPatch is the current patch revision.
	CurrentPatch uint16 = 0
)

// CurrentVersion is the protocol version produced by this OTGH build.
//
// Version values are immutable value types, so exposing this package variable
// does not expose mutable protocol state.
var CurrentVersion = Version{
	Major: CurrentMajor,
	Minor: CurrentMinor,
	Patch: CurrentPatch,
}

// MinimumSupportedVersion is the oldest protocol version this implementation
// explicitly supports.
//
// As the protocol evolves this can move independently from CurrentVersion.
var MinimumSupportedVersion = Version{
	Major: 1,
	Minor: 0,
	Patch: 0,
}

// String returns the canonical dotted version representation.
func (v Version) String() string {
	return fmt.Sprintf(
		"%d.%d.%d",
		v.Major,
		v.Minor,
		v.Patch,
	)
}

// Validate verifies that a Version represents a usable OTGH protocol version.
func (v Version) Validate() error {
	if v.Major == 0 {
		return fmt.Errorf(
			"%w: major version must be greater than zero",
			ErrInvalidVersion,
		)
	}

	return nil
}

// IsZero reports whether all version components are zero.
func (v Version) IsZero() bool {
	return v.Major == 0 &&
		v.Minor == 0 &&
		v.Patch == 0
}

// Equal reports exact version equality.
func (v Version) Equal(other Version) bool {
	return v == other
}

// Compare compares two versions.
//
// It returns:
//
//   - -1 when v is older than other;
//   -  0 when they are equal;
//   -  1 when v is newer than other.
func (v Version) Compare(other Version) int {
	switch {
	case v.Major < other.Major:
		return -1

	case v.Major > other.Major:
		return 1

	case v.Minor < other.Minor:
		return -1

	case v.Minor > other.Minor:
		return 1

	case v.Patch < other.Patch:
		return -1

	case v.Patch > other.Patch:
		return 1

	default:
		return 0
	}
}

// Before reports whether v is older than other.
func (v Version) Before(other Version) bool {
	return v.Compare(other) < 0
}

// After reports whether v is newer than other.
func (v Version) After(other Version) bool {
	return v.Compare(other) > 0
}

// CompatibleWith reports whether v and other may communicate.
//
// OTGH v1 compatibility policy:
//
//   - both versions must be valid;
//   - major versions must match;
//   - patch differences are compatible;
//   - minor versions within the same major family are considered capable of
//     negotiation.
//
// Feature-level negotiation will later be handled by handshake.go rather than
// assuming that every feature introduced by a newer minor version exists on
// the older peer.
func (v Version) CompatibleWith(other Version) bool {
	if v.Validate() != nil || other.Validate() != nil {
		return false
	}

	return v.Major == other.Major
}

// SupportedByCurrent reports whether version is within the protocol family
// supported by this implementation.
//
// A version newer than CurrentVersion may share the same major version and
// therefore be wire-compatible at the version level, but feature negotiation
// must still prevent unsupported capabilities from being used.
func (v Version) SupportedByCurrent() bool {
	if v.Validate() != nil {
		return false
	}

	if !v.CompatibleWith(CurrentVersion) {
		return false
	}

	if v.Before(MinimumSupportedVersion) {
		return false
	}

	return true
}

// ParseVersion parses the canonical:
//
//	major.minor.patch
//
// representation.
//
// Prefixes such as "v1.0.0", missing components, negative values, whitespace
// inside the version, and additional components are rejected deliberately so
// version negotiation has one unambiguous textual form.
func ParseVersion(value string) (Version, error) {
	value = strings.TrimSpace(value)

	if value == "" {
		return Version{}, fmt.Errorf(
			"%w: version is empty",
			ErrInvalidVersion,
		)
	}

	parts := strings.Split(value, ".")

	if len(parts) != 3 {
		return Version{}, fmt.Errorf(
			"%w: expected major.minor.patch, got %q",
			ErrInvalidVersion,
			value,
		)
	}

	major, err := parseVersionComponent(
		parts[0],
		"major",
	)
	if err != nil {
		return Version{}, err
	}

	minor, err := parseVersionComponent(
		parts[1],
		"minor",
	)
	if err != nil {
		return Version{}, err
	}

	patch, err := parseVersionComponent(
		parts[2],
		"patch",
	)
	if err != nil {
		return Version{}, err
	}

	version := Version{
		Major: major,
		Minor: minor,
		Patch: patch,
	}

	if err := version.Validate(); err != nil {
		return Version{}, err
	}

	return version, nil
}

// NegotiateVersion selects the newest compatible version shared by local and
// remote peers.
//
// Each side provides the concrete protocol versions it can speak.
//
// Negotiation does not rely on input order and rejects invalid versions rather
// than silently ignoring malformed protocol declarations.
func NegotiateVersion(
	local []Version,
	remote []Version,
) (Version, error) {
	if len(local) == 0 {
		return Version{}, fmt.Errorf(
			"%w: local version set is empty",
			ErrNoCompatibleVersion,
		)
	}

	if len(remote) == 0 {
		return Version{}, fmt.Errorf(
			"%w: remote version set is empty",
			ErrNoCompatibleVersion,
		)
	}

	for i, version := range local {
		if err := version.Validate(); err != nil {
			return Version{}, fmt.Errorf(
				"local version %d: %w",
				i,
				err,
			)
		}
	}

	for i, version := range remote {
		if err := version.Validate(); err != nil {
			return Version{}, fmt.Errorf(
				"remote version %d: %w",
				i,
				err,
			)
		}
	}

	var (
		selected Version
		found    bool
	)

	// Prefer exact shared versions. This ensures that both peers have
	// explicitly declared support for the negotiated wire revision.
	for _, localVersion := range local {
		for _, remoteVersion := range remote {
			if !localVersion.Equal(remoteVersion) {
				continue
			}

			if !found || localVersion.After(selected) {
				selected = localVersion
				found = true
			}
		}
	}

	if found {
		return selected, nil
	}

	return Version{}, fmt.Errorf(
		"%w",
		ErrNoCompatibleVersion,
	)
}

// HighestVersion returns the newest version from a non-empty set.
func HighestVersion(
	versions []Version,
) (Version, error) {
	if len(versions) == 0 {
		return Version{}, fmt.Errorf(
			"%w: version set is empty",
			ErrInvalidVersion,
		)
	}

	var highest Version

	for i, version := range versions {
		if err := version.Validate(); err != nil {
			return Version{}, fmt.Errorf(
				"version %d: %w",
				i,
				err,
			)
		}

		if i == 0 || version.After(highest) {
			highest = version
		}
	}

	return highest, nil
}

// SupportedVersions returns the concrete protocol versions this build
// currently advertises.
//
// For the initial protocol revision there is only one. Future builds may
// advertise multiple versions during compatibility migrations.
func SupportedVersions() []Version {
	return []Version{
		CurrentVersion,
	}
}

func parseVersionComponent(
	value string,
	name string,
) (uint16, error) {
	if value == "" {
		return 0, fmt.Errorf(
			"%w: %s component is empty",
			ErrInvalidVersion,
			name,
		)
	}

	// Reject signs, whitespace, and other non-decimal syntax before calling
	// ParseUint so the canonical representation remains strict.
	for _, character := range value {
		if character < '0' || character > '9' {
			return 0, fmt.Errorf(
				"%w: %s component %q is not an unsigned decimal integer",
				ErrInvalidVersion,
				name,
				value,
			)
		}
	}

	parsed, err := strconv.ParseUint(
		value,
		10,
		16,
	)
	if err != nil {
		return 0, fmt.Errorf(
			"%w: invalid %s component %q: %v",
			ErrInvalidVersion,
			name,
			value,
			err,
		)
	}

	return uint16(parsed), nil
}
