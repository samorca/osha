package export

import (
	"errors"
	"fmt"
	"strings"
	"sync"
)

// ErrNoRoute indicates that no destination could be selected for a record.
var ErrNoRoute = errors.New("no route found for record")

// ErrDestinationNotFound indicates that a requested destination is not
// registered with the router.
var ErrDestinationNotFound = errors.New("destination not found")

// Route describes a routing rule.
//
// A rule may match a record by type, minimum priority, tenant, and arbitrary
// metadata. Empty matching fields act as wildcards.
//
// DestinationIDs contains the destinations that receive a matching record.
// A single record may therefore fan out to multiple destinations.
type Route struct {
	Name string

	RecordTypes []RecordType
	MinPriority Priority
	TenantID    TenantID

	Metadata map[string]string

	DestinationIDs []DestinationID
}

// Validate verifies that a route is usable.
func (r Route) Validate() error {
	if strings.TrimSpace(r.Name) == "" {
		return errors.New("route name is required")
	}

	if !r.MinPriority.Valid() {
		return fmt.Errorf("route %q has invalid minimum priority %d", r.Name, r.MinPriority)
	}

	if len(r.DestinationIDs) == 0 {
		return fmt.Errorf("route %q must contain at least one destination", r.Name)
	}

	for i, recordType := range r.RecordTypes {
		if !recordType.Valid() {
			return fmt.Errorf(
				"route %q contains invalid record type at index %d: %q",
				r.Name,
				i,
				recordType,
			)
		}
	}

	seen := make(map[DestinationID]struct{}, len(r.DestinationIDs))

	for i, destinationID := range r.DestinationIDs {
		if strings.TrimSpace(string(destinationID)) == "" {
			return fmt.Errorf(
				"route %q contains empty destination id at index %d",
				r.Name,
				i,
			)
		}

		if _, exists := seen[destinationID]; exists {
			return fmt.Errorf(
				"route %q contains duplicate destination %q",
				r.Name,
				destinationID,
			)
		}

		seen[destinationID] = struct{}{}
	}

	return nil
}

// Match reports whether this route applies to the supplied record.
func (r Route) Match(record Record) bool {
	if len(r.RecordTypes) > 0 && !containsRecordType(r.RecordTypes, record.Type) {
		return false
	}

	if record.Priority < r.MinPriority {
		return false
	}

	if r.TenantID != "" && r.TenantID != record.TenantID {
		return false
	}

	for key, expected := range r.Metadata {
		actual, exists := record.Metadata[key]
		if !exists || actual != expected {
			return false
		}
	}

	return true
}

// RouteResult describes the destinations selected by the router.
//
// Multiple routes may match the same record. Destinations are de-duplicated
// while preserving the order in which they were first selected.
type RouteResult struct {
	RecordID     RecordID
	Destinations []Destination
	MatchedRules []string
}

// Router provides thread-safe routing for OTGH records.
//
// Router owns routing policy only. It does not perform delivery, retry,
// batching, or acknowledgement. Those responsibilities belong to the
// corresponding export components.
type Router struct {
	mu sync.RWMutex

	destinations map[DestinationID]Destination
	routes       []Route
}

// NewRouter creates an empty router.
func NewRouter() *Router {
	return &Router{
		destinations: make(map[DestinationID]Destination),
		routes:       make([]Route, 0),
	}
}

// RegisterDestination adds or replaces a destination.
//
// Replacing an existing destination with the same ID is intentional. This
// allows configuration reload code to update endpoints without rebuilding
// routing rules that reference the destination ID.
func (r *Router) RegisterDestination(destination Destination) error {
	if r == nil {
		return errors.New("router is nil")
	}

	if err := destination.Validate(); err != nil {
		return fmt.Errorf("register destination: %w", err)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if r.destinations == nil {
		r.destinations = make(map[DestinationID]Destination)
	}

	r.destinations[destination.ID] = cloneDestination(destination)

	return nil
}

// RemoveDestination removes a destination.
//
// The router intentionally does not rewrite routes. A stale route referencing
// the removed destination will fail validation when routes are replaced and
// will be ignored during runtime resolution.
func (r *Router) RemoveDestination(id DestinationID) bool {
	if r == nil {
		return false
	}

	if strings.TrimSpace(string(id)) == "" {
		return false
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if r.destinations == nil {
		return false
	}

	if _, exists := r.destinations[id]; !exists {
		return false
	}

	delete(r.destinations, id)

	return true
}

// Destination returns a defensive copy of a registered destination.
func (r *Router) Destination(id DestinationID) (Destination, bool) {
	if r == nil {
		return Destination{}, false
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	destination, exists := r.destinations[id]
	if !exists {
		return Destination{}, false
	}

	return cloneDestination(destination), true
}

// Destinations returns a snapshot of all registered destinations.
func (r *Router) Destinations() []Destination {
	if r == nil {
		return nil
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make([]Destination, 0, len(r.destinations))

	for _, destination := range r.destinations {
		result = append(result, cloneDestination(destination))
	}

	return result
}

// AddRoute validates and appends one routing rule.
//
// All referenced destinations must already be registered.
func (r *Router) AddRoute(route Route) error {
	if r == nil {
		return errors.New("router is nil")
	}

	if err := route.Validate(); err != nil {
		return fmt.Errorf("add route: %w", err)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.validateRouteDestinationsLocked(route); err != nil {
		return err
	}

	r.routes = append(r.routes, cloneRoute(route))

	return nil
}

// ReplaceRoutes atomically replaces the complete routing table.
//
// Validation occurs before the current routing table is modified. If any new
// route is invalid, the existing routing table remains untouched.
func (r *Router) ReplaceRoutes(routes []Route) error {
	if r == nil {
		return errors.New("router is nil")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	validated := make([]Route, 0, len(routes))
	names := make(map[string]struct{}, len(routes))

	for i, route := range routes {
		if err := route.Validate(); err != nil {
			return fmt.Errorf("route %d: %w", i, err)
		}

		if _, exists := names[route.Name]; exists {
			return fmt.Errorf("duplicate route name %q", route.Name)
		}

		if err := r.validateRouteDestinationsLocked(route); err != nil {
			return fmt.Errorf("route %q: %w", route.Name, err)
		}

		names[route.Name] = struct{}{}
		validated = append(validated, cloneRoute(route))
	}

	r.routes = validated

	return nil
}

// Routes returns a defensive snapshot of the current routing table.
func (r *Router) Routes() []Route {
	if r == nil {
		return nil
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make([]Route, 0, len(r.routes))

	for _, route := range r.routes {
		result = append(result, cloneRoute(route))
	}

	return result
}

// Route resolves a record into one or more destinations.
func (r *Router) Route(record Record) (RouteResult, error) {
	if r == nil {
		return RouteResult{}, errors.New("router is nil")
	}

	if err := record.Validate(); err != nil {
		return RouteResult{}, fmt.Errorf("route record: %w", err)
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	result := RouteResult{
		RecordID:     record.ID,
		Destinations: make([]Destination, 0),
		MatchedRules: make([]string, 0),
	}

	selected := make(map[DestinationID]struct{})

	for _, route := range r.routes {
		if !route.Match(record) {
			continue
		}

		result.MatchedRules = append(result.MatchedRules, route.Name)

		for _, destinationID := range route.DestinationIDs {
			if _, alreadySelected := selected[destinationID]; alreadySelected {
				continue
			}

			destination, exists := r.destinations[destinationID]
			if !exists {
				// Configuration can change at runtime. A removed destination
				// must not make unrelated matching destinations unusable.
				continue
			}

			selected[destinationID] = struct{}{}
			result.Destinations = append(
				result.Destinations,
				cloneDestination(destination),
			)
		}
	}

	if len(result.Destinations) == 0 {
		return result, fmt.Errorf("%w: record %q", ErrNoRoute, record.ID)
	}

	return result, nil
}

// Validate verifies that the complete routing configuration is internally
// consistent.
func (r *Router) Validate() error {
	if r == nil {
		return errors.New("router is nil")
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	for id, destination := range r.destinations {
		if err := destination.Validate(); err != nil {
			return fmt.Errorf("destination %q: %w", id, err)
		}
	}

	names := make(map[string]struct{}, len(r.routes))

	for i, route := range r.routes {
		if err := route.Validate(); err != nil {
			return fmt.Errorf("route %d: %w", i, err)
		}

		if _, exists := names[route.Name]; exists {
			return fmt.Errorf("duplicate route name %q", route.Name)
		}

		if err := r.validateRouteDestinationsLocked(route); err != nil {
			return fmt.Errorf("route %q: %w", route.Name, err)
		}

		names[route.Name] = struct{}{}
	}

	return nil
}

func (r *Router) validateRouteDestinationsLocked(route Route) error {
	for _, destinationID := range route.DestinationIDs {
		if _, exists := r.destinations[destinationID]; !exists {
			return fmt.Errorf(
				"%w: %q",
				ErrDestinationNotFound,
				destinationID,
			)
		}
	}

	return nil
}

func containsRecordType(types []RecordType, target RecordType) bool {
	for _, recordType := range types {
		if recordType == target {
			return true
		}
	}

	return false
}

func cloneDestination(destination Destination) Destination {
	cloned := destination

	if destination.Metadata != nil {
		cloned.Metadata = make(map[string]string, len(destination.Metadata))

		for key, value := range destination.Metadata {
			cloned.Metadata[key] = value
		}
	}

	return cloned
}

func cloneRoute(route Route) Route {
	cloned := route

	if route.RecordTypes != nil {
		cloned.RecordTypes = append([]RecordType(nil), route.RecordTypes...)
	}

	if route.DestinationIDs != nil {
		cloned.DestinationIDs = append(
			[]DestinationID(nil),
			route.DestinationIDs...,
		)
	}

	if route.Metadata != nil {
		cloned.Metadata = make(map[string]string, len(route.Metadata))

		for key, value := range route.Metadata {
			cloned.Metadata[key] = value
		}
	}

	return cloned
}
