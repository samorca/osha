package protocol

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

// AckStatus represents the receiving peer's disposition of a message or
// logical batch.
type AckStatus string

const (
	AckStatusAccepted AckStatus = "accepted"
	AckStatusPartial  AckStatus = "partial"
	AckStatusRejected AckStatus = "rejected"
	AckStatusDeferred AckStatus = "deferred"
)

// Valid reports whether the acknowledgement status is recognised.
func (s AckStatus) Valid() bool {
	switch s {
	case AckStatusAccepted,
		AckStatusPartial,
		AckStatusRejected,
		AckStatusDeferred:
		return true

	default:
		return false
	}
}

// Terminal reports whether this acknowledgement represents a completed
// disposition.
//
// Accepted and Rejected are terminal from the perspective of the current
// delivery operation. Partial and Deferred may require additional handling.
func (s AckStatus) Terminal() bool {
	switch s {
	case AckStatusAccepted,
		AckStatusRejected:
		return true

	default:
		return false
	}
}

// AckItemStatus describes the disposition of one record inside a logical
// batch acknowledgement.
type AckItemStatus string

const (
	AckItemAccepted AckItemStatus = "accepted"
	AckItemRejected AckItemStatus = "rejected"
	AckItemDeferred AckItemStatus = "deferred"
)

// Valid reports whether the item acknowledgement status is recognised.
func (s AckItemStatus) Valid() bool {
	switch s {
	case AckItemAccepted,
		AckItemRejected,
		AckItemDeferred:
		return true

	default:
		return false
	}
}

// AckItem describes the receiving peer's result for one logical record.
//
// Record-level acknowledgement is optional. It becomes useful when a batch is
// only partially accepted and the sender needs to know exactly which records
// require further handling.
type AckItem struct {
	RecordID string `json:"record_id"`

	Status AckItemStatus `json:"status"`

	Error *ProtocolError `json:"error,omitempty"`
}

// Validate verifies an item acknowledgement.
func (i AckItem) Validate() error {
	if strings.TrimSpace(i.RecordID) == "" {
		return errors.New(
			"acknowledgement item record_id is required",
		)
	}

	if !i.Status.Valid() {
		return fmt.Errorf(
			"invalid acknowledgement item status %q",
			i.Status,
		)
	}

	switch i.Status {
	case AckItemAccepted:
		if i.Error != nil {
			return errors.New(
				"accepted acknowledgement item cannot contain an error",
			)
		}

	case AckItemRejected:
		if i.Error == nil {
			return errors.New(
				"rejected acknowledgement item requires an error",
			)
		}

	case AckItemDeferred:
		// A deferred item may contain an explanatory error, but an error is
		// not mandatory.
	}

	if i.Error != nil {
		if err := i.Error.Validate(); err != nil {
			return fmt.Errorf(
				"acknowledgement item error: %w",
				err,
			)
		}
	}

	return nil
}

// Acknowledgement is the wire-level response confirming how a peer handled a
// protocol message.
//
// This type belongs to the OTGH protocol itself. It is different from the
// process-local acknowledgement manager in internal/export.
type Acknowledgement struct {
	// MessageID identifies the original Envelope being acknowledged.
	MessageID string `json:"message_id"`

	// CorrelationID preserves the original exchange correlation when
	// available.
	CorrelationID string `json:"correlation_id,omitempty"`

	// SessionID identifies the negotiated protocol session.
	SessionID string `json:"session_id"`

	// Sequence is the sequence number from the original message.
	Sequence uint64 `json:"sequence,omitempty"`

	Status AckStatus `json:"status"`

	// TotalRecords is the number of logical records represented by the
	// original message/batch.
	TotalRecords uint32 `json:"total_records"`

	AcceptedRecords uint32 `json:"accepted_records"`
	RejectedRecords uint32 `json:"rejected_records"`
	DeferredRecords uint32 `json:"deferred_records"`

	// Retryable is a peer hint indicating that another delivery attempt may
	// succeed.
	//
	// The sender's retry policy still controls whether a retry actually
	// occurs.
	Retryable bool `json:"retryable"`

	// RetryAfterMillis optionally suggests how long the sender should wait
	// before retrying.
	//
	// Zero means no remote delay suggestion was provided.
	RetryAfterMillis uint64 `json:"retry_after_millis,omitempty"`

	// Error describes a message-level failure.
	Error *ProtocolError `json:"error,omitempty"`

	// Items contains optional per-record results.
	Items []AckItem `json:"items,omitempty"`

	ReceivedAt time.Time `json:"received_at"`

	Metadata map[string]string `json:"metadata,omitempty"`
}

// Validate verifies acknowledgement consistency.
func (a Acknowledgement) Validate() error {
	if strings.TrimSpace(a.MessageID) == "" {
		return errors.New(
			"acknowledgement message_id is required",
		)
	}

	if strings.TrimSpace(a.SessionID) == "" {
		return errors.New(
			"acknowledgement session_id is required",
		)
	}

	if !a.Status.Valid() {
		return fmt.Errorf(
			"invalid acknowledgement status %q",
			a.Status,
		)
	}

	if a.TotalRecords == 0 {
		return errors.New(
			"acknowledgement total_records must be greater than zero",
		)
	}

	accounted := uint64(a.AcceptedRecords) +
		uint64(a.RejectedRecords) +
		uint64(a.DeferredRecords)

	if accounted > uint64(a.TotalRecords) {
		return fmt.Errorf(
			"acknowledgement accounts for %d records but total_records is %d",
			accounted,
			a.TotalRecords,
		)
	}

	if a.ReceivedAt.IsZero() {
		return errors.New(
			"acknowledgement received_at is required",
		)
	}

	if a.Error != nil {
		if err := a.Error.Validate(); err != nil {
			return fmt.Errorf(
				"acknowledgement error: %w",
				err,
			)
		}
	}

	if err := validateAckItems(
		a.Items,
		a.TotalRecords,
	); err != nil {
		return err
	}

	switch a.Status {
	case AckStatusAccepted:
		if a.AcceptedRecords != a.TotalRecords {
			return errors.New(
				"accepted acknowledgement must accept every record",
			)
		}

		if a.RejectedRecords != 0 ||
			a.DeferredRecords != 0 {
			return errors.New(
				"accepted acknowledgement cannot contain rejected or deferred records",
			)
		}

		if a.Retryable {
			return errors.New(
				"accepted acknowledgement cannot be retryable",
			)
		}

		if a.RetryAfterMillis != 0 {
			return errors.New(
				"accepted acknowledgement cannot specify retry_after_millis",
			)
		}

		if a.Error != nil {
			return errors.New(
				"accepted acknowledgement cannot contain an error",
			)
		}

	case AckStatusRejected:
		if a.RejectedRecords != a.TotalRecords {
			return errors.New(
				"rejected acknowledgement must reject every record",
			)
		}

		if a.AcceptedRecords != 0 ||
			a.DeferredRecords != 0 {
			return errors.New(
				"rejected acknowledgement cannot contain accepted or deferred records",
			)
		}

		if a.Error == nil {
			return errors.New(
				"rejected acknowledgement requires an error",
			)
		}

	case AckStatusDeferred:
		if a.DeferredRecords != a.TotalRecords {
			return errors.New(
				"deferred acknowledgement must defer every record",
			)
		}

		if a.AcceptedRecords != 0 ||
			a.RejectedRecords != 0 {
			return errors.New(
				"deferred acknowledgement cannot contain accepted or rejected records",
			)
		}

		if !a.Retryable {
			return errors.New(
				"deferred acknowledgement must be retryable",
			)
		}

	case AckStatusPartial:
		if a.AcceptedRecords == a.TotalRecords {
			return errors.New(
				"partial acknowledgement cannot accept every record",
			)
		}

		if accounted != uint64(a.TotalRecords) {
			return errors.New(
				"partial acknowledgement must account for every record",
			)
		}

		if a.AcceptedRecords == 0 &&
			a.RejectedRecords == 0 &&
			a.DeferredRecords == 0 {
			return errors.New(
				"partial acknowledgement contains no record dispositions",
			)
		}
	}

	if a.RetryAfterMillis > 0 && !a.Retryable {
		return errors.New(
			"retry_after_millis requires retryable acknowledgement",
		)
	}

	return nil
}

// Complete reports whether every logical record was accepted.
func (a Acknowledgement) Complete() bool {
	return a.Status == AckStatusAccepted &&
		a.AcceptedRecords == a.TotalRecords &&
		a.RejectedRecords == 0 &&
		a.DeferredRecords == 0
}

// OutstandingRecords returns the number of records not accounted for by the
// acknowledgement.
//
// Valid accepted/rejected/deferred acknowledgements normally account for all
// records. This method is still useful before full validation or while
// constructing an acknowledgement.
func (a Acknowledgement) OutstandingRecords() uint32 {
	accounted := uint64(a.AcceptedRecords) +
		uint64(a.RejectedRecords) +
		uint64(a.DeferredRecords)

	if accounted >= uint64(a.TotalRecords) {
		return 0
	}

	return uint32(
		uint64(a.TotalRecords) - accounted,
	)
}

// RetryAfter converts RetryAfterMillis into a duration.
func (a Acknowledgement) RetryAfter() time.Duration {
	if a.RetryAfterMillis == 0 {
		return 0
	}

	const maxDurationMillis = uint64(
		(^uint64(0) >> 1) / uint64(time.Millisecond),
	)

	if a.RetryAfterMillis > maxDurationMillis {
		return time.Duration(^uint64(0) >> 1)
	}

	return time.Duration(a.RetryAfterMillis) *
		time.Millisecond
}

// Clone returns a defensive copy.
func (a Acknowledgement) Clone() Acknowledgement {
	cloned := a

	if a.Metadata != nil {
		cloned.Metadata = cloneStringMap(
			a.Metadata,
		)
	}

	if a.Error != nil {
		errorCopy := *a.Error

		if a.Error.Details != nil {
			errorCopy.Details = cloneStringMap(
				a.Error.Details,
			)
		}

		cloned.Error = &errorCopy
	}

	if a.Items != nil {
		cloned.Items = make(
			[]AckItem,
			len(a.Items),
		)

		for index := range a.Items {
			cloned.Items[index] =
				cloneAckItem(a.Items[index])
		}
	}

	return cloned
}

// NewAcceptedAcknowledgement constructs a complete successful
// acknowledgement.
func NewAcceptedAcknowledgement(
	messageID string,
	sessionID string,
	correlationID string,
	sequence uint64,
	totalRecords uint32,
) (Acknowledgement, error) {
	if totalRecords == 0 {
		return Acknowledgement{}, errors.New(
			"total records must be greater than zero",
		)
	}

	ack := Acknowledgement{
		MessageID: strings.TrimSpace(
			messageID,
		),
		SessionID: strings.TrimSpace(
			sessionID,
		),
		CorrelationID: strings.TrimSpace(
			correlationID,
		),
		Sequence: sequence,

		Status: AckStatusAccepted,

		TotalRecords:    totalRecords,
		AcceptedRecords: totalRecords,

		ReceivedAt: time.Now().UTC(),
	}

	if err := ack.Validate(); err != nil {
		return Acknowledgement{}, err
	}

	return ack, nil
}

// NewRejectedAcknowledgement constructs a complete rejection.
func NewRejectedAcknowledgement(
	messageID string,
	sessionID string,
	correlationID string,
	sequence uint64,
	totalRecords uint32,
	protocolError *ProtocolError,
	retryable bool,
	retryAfter time.Duration,
) (Acknowledgement, error) {
	if totalRecords == 0 {
		return Acknowledgement{}, errors.New(
			"total records must be greater than zero",
		)
	}

	if protocolError == nil {
		return Acknowledgement{}, errors.New(
			"rejected acknowledgement requires protocol error",
		)
	}

	if retryAfter < 0 {
		return Acknowledgement{}, errors.New(
			"retry delay cannot be negative",
		)
	}

	retryMillis, err := durationToMilliseconds(
		retryAfter,
	)
	if err != nil {
		return Acknowledgement{}, err
	}

	ack := Acknowledgement{
		MessageID: strings.TrimSpace(
			messageID,
		),
		SessionID: strings.TrimSpace(
			sessionID,
		),
		CorrelationID: strings.TrimSpace(
			correlationID,
		),
		Sequence: sequence,

		Status: AckStatusRejected,

		TotalRecords:    totalRecords,
		RejectedRecords: totalRecords,

		Retryable: retryable,

		RetryAfterMillis: retryMillis,

		Error: cloneProtocolError(
			protocolError,
		),

		ReceivedAt: time.Now().UTC(),
	}

	if err := ack.Validate(); err != nil {
		return Acknowledgement{}, err
	}

	return ack, nil
}

// NewDeferredAcknowledgement constructs a temporary deferral.
//
// The receiver is explicitly indicating that no records were accepted or
// rejected and that the sender may retry later.
func NewDeferredAcknowledgement(
	messageID string,
	sessionID string,
	correlationID string,
	sequence uint64,
	totalRecords uint32,
	retryAfter time.Duration,
	protocolError *ProtocolError,
) (Acknowledgement, error) {
	if totalRecords == 0 {
		return Acknowledgement{}, errors.New(
			"total records must be greater than zero",
		)
	}

	if retryAfter < 0 {
		return Acknowledgement{}, errors.New(
			"retry delay cannot be negative",
		)
	}

	retryMillis, err := durationToMilliseconds(
		retryAfter,
	)
	if err != nil {
		return Acknowledgement{}, err
	}

	ack := Acknowledgement{
		MessageID: strings.TrimSpace(
			messageID,
		),
		SessionID: strings.TrimSpace(
			sessionID,
		),
		CorrelationID: strings.TrimSpace(
			correlationID,
		),
		Sequence: sequence,

		Status: AckStatusDeferred,

		TotalRecords:    totalRecords,
		DeferredRecords: totalRecords,

		Retryable: true,

		RetryAfterMillis: retryMillis,

		Error: cloneProtocolError(
			protocolError,
		),

		ReceivedAt: time.Now().UTC(),
	}

	if err := ack.Validate(); err != nil {
		return Acknowledgement{}, err
	}

	return ack, nil
}

// NewPartialAcknowledgement constructs a record-level partial result.
//
// Items must collectively describe every record in the logical batch.
func NewPartialAcknowledgement(
	messageID string,
	sessionID string,
	correlationID string,
	sequence uint64,
	totalRecords uint32,
	items []AckItem,
	retryable bool,
	retryAfter time.Duration,
) (Acknowledgement, error) {
	if totalRecords == 0 {
		return Acknowledgement{}, errors.New(
			"total records must be greater than zero",
		)
	}

	if len(items) == 0 {
		return Acknowledgement{}, errors.New(
			"partial acknowledgement requires item results",
		)
	}

	if retryAfter < 0 {
		return Acknowledgement{}, errors.New(
			"retry delay cannot be negative",
		)
	}

	if retryAfter > 0 && !retryable {
		return Acknowledgement{}, errors.New(
			"retry delay requires retryable acknowledgement",
		)
	}

	var (
		accepted uint32
		rejected uint32
		deferred uint32
	)

	clonedItems := make(
		[]AckItem,
		len(items),
	)

	for index, item := range items {
		if err := item.Validate(); err != nil {
			return Acknowledgement{}, fmt.Errorf(
				"acknowledgement item %d: %w",
				index,
				err,
			)
		}

		clonedItems[index] = cloneAckItem(
			item,
		)

		switch item.Status {
		case AckItemAccepted:
			accepted++

		case AckItemRejected:
			rejected++

		case AckItemDeferred:
			deferred++
		}
	}

	retryMillis, err := durationToMilliseconds(
		retryAfter,
	)
	if err != nil {
		return Acknowledgement{}, err
	}

	ack := Acknowledgement{
		MessageID: strings.TrimSpace(
			messageID,
		),
		SessionID: strings.TrimSpace(
			sessionID,
		),
		CorrelationID: strings.TrimSpace(
			correlationID,
		),
		Sequence: sequence,

		Status: AckStatusPartial,

		TotalRecords: totalRecords,

		AcceptedRecords: accepted,
		RejectedRecords: rejected,
		DeferredRecords: deferred,

		Retryable: retryable,

		RetryAfterMillis: retryMillis,

		Items: clonedItems,

		ReceivedAt: time.Now().UTC(),
	}

	if err := ack.Validate(); err != nil {
		return Acknowledgement{}, err
	}

	return ack, nil
}

func validateAckItems(
	items []AckItem,
	totalRecords uint32,
) error {
	if len(items) == 0 {
		return nil
	}

	if uint64(len(items)) >
		uint64(totalRecords) {
		return fmt.Errorf(
			"acknowledgement contains %d item results but total_records is %d",
			len(items),
			totalRecords,
		)
	}

	seen := make(
		map[string]struct{},
		len(items),
	)

	for index, item := range items {
		if err := item.Validate(); err != nil {
			return fmt.Errorf(
				"acknowledgement item %d: %w",
				index,
				err,
			)
		}

		recordID := strings.TrimSpace(
			item.RecordID,
		)

		if _, exists := seen[recordID]; exists {
			return fmt.Errorf(
				"duplicate acknowledgement record_id %q",
				recordID,
			)
		}

		seen[recordID] = struct{}{}
	}

	return nil
}

func cloneAckItem(
	item AckItem,
) AckItem {
	cloned := item

	if item.Error != nil {
		cloned.Error = cloneProtocolError(
			item.Error,
		)
	}

	return cloned
}

func cloneProtocolError(
	input *ProtocolError,
) *ProtocolError {
	if input == nil {
		return nil
	}

	cloned := *input

	if input.Details != nil {
		cloned.Details = cloneStringMap(
			input.Details,
		)
	}

	return &cloned
}

func durationToMilliseconds(
	duration time.Duration,
) (uint64, error) {
	if duration < 0 {
		return 0, errors.New(
			"duration cannot be negative",
		)
	}

	if duration == 0 {
		return 0, nil
	}

	milliseconds := duration.Milliseconds()

	if milliseconds <= 0 {
		// Preserve a positive delay smaller than one millisecond rather than
		// accidentally converting it into "no delay suggestion".
		return 1, nil
	}

	return uint64(milliseconds), nil
}
