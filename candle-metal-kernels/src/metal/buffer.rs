use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{MTLBuffer, MTLResource};
use std::{collections::HashMap, sync::Arc};

pub type MetalResource = ProtocolObject<dyn MTLResource>;
pub type MTLResourceOptions = objc2_metal::MTLResourceOptions;

#[derive(Clone, Debug, Hash, PartialEq)]
pub struct Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,

    /// Byte offset this handle addresses within `raw`.
    ///
    /// Zero for every buffer candle allocates today, which is why this is not a
    /// behaviour change: an ordinary allocation owns its whole `MTLBuffer` and
    /// starts at 0. It is nonzero only for an activation-arena slot, which is a
    /// *region* of one shared allocation rather than an allocation of its own
    /// (`DESIGN.md` §9.2).
    ///
    /// # Why the offset lives here rather than at the call sites
    ///
    /// The arena binds every activation as `base + offset`. Threading that
    /// offset outward would mean touching the ~51 kernel entry points that take
    /// `output: &Buffer` and the 62 `buffer_o` call sites in the Metal backend
    /// -- and under `HazardTrackingModeUntracked` (§3.5) every one of those is
    /// an opportunity for a silent wrong-offset bind, which is precisely the
    /// failure class §9.3 says rests entirely on our offset arithmetic.
    ///
    /// Carrying it in the handle instead means the addition happens in exactly
    /// two places -- `set_input_buffer` and `set_output_buffer`, the choke point
    /// every binding already funnels through. That is the same argument §11.1b
    /// makes for putting the executor seam at dispatch submission rather than at
    /// the kernel entry points: one place to review, not fifty.
    base_offset: usize,

    /// Bytes this handle addresses, when it is a view.
    ///
    /// `None` means "all of `raw`", which is every ordinary allocation. A view
    /// carries its slot size so `length()` cannot report the whole arena.
    length: Option<usize>,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLBuffer>>) -> Buffer {
        Buffer {
            raw,
            base_offset: 0,
            length: None,
        }
    }

    /// A handle addressing `base_offset..base_offset + length` of this buffer.
    ///
    /// Shares the underlying `MTLBuffer` -- this is a view, not a copy, and
    /// nothing is allocated. Used by the activation arena to hand out slots.
    ///
    /// The shared `MTLBuffer` identity is deliberate on two counts. It is what
    /// makes buffer identity *stable* at a dispatch position (§11.1a.1's 674),
    /// since every slot presents the same allocation. And it makes the
    /// encoder's hazard tracking, which keys on the raw pointer, treat the whole
    /// arena as one resource -- so a read-after-write on any slot orders against
    /// a write to any other. That is conservative rather than unsound: Metal
    /// barriers are resource-granular anyway (§3.5), so a finer key could not
    /// have expressed a narrower edge.
    pub fn view(&self, base_offset: usize, length: usize) -> Buffer {
        debug_assert!(
            base_offset + length <= self.length(),
            "arena view {base_offset}..{} runs past its {} byte parent",
            base_offset + length,
            self.length()
        );
        Buffer {
            raw: self.raw.clone(),
            base_offset: self.base_offset + base_offset,
            length: Some(length),
        }
    }

    /// Byte offset this handle addresses within the underlying `MTLBuffer`.
    pub fn base_offset(&self) -> usize {
        self.base_offset
    }

    pub fn contents(&self) -> *mut u8 {
        self.data()
    }

    /// CPU pointer to the first byte *this handle addresses*.
    ///
    /// Offset by `base_offset`, so a view reads and writes its own region
    /// rather than the parent's start. Keeping the whole `Buffer` API
    /// consistent about what "this buffer" means is what stops a view from
    /// being a silent-corruption hazard: `read_to_vec` and every other CPU-side
    /// consumer takes this pointer without knowing whether it holds a view.
    ///
    /// Null for a Private-storage allocation, which has no CPU mapping at all
    /// -- unchanged by the offset, since `null + n` is still not readable and
    /// callers already assert on it.
    pub fn data(&self) -> *mut u8 {
        use objc2_metal::MTLBuffer as _;
        let base = self.as_ref().contents().as_ptr() as *mut u8;
        if base.is_null() {
            return base;
        }
        // SAFETY: `base_offset` is within the parent allocation by construction
        // -- `view` is only called by the arena, with an offset and length that
        // the plan has already fitted inside the arena buffer.
        unsafe { base.add(self.base_offset) }
    }

    /// Get the raw pointer to the underlying Metal buffer object.
    /// Used for dependency tracking in the compute encoder.
    pub(crate) fn raw_ptr(&self) -> *const ProtocolObject<dyn MTLBuffer> {
        Retained::as_ptr(&self.raw)
    }

    /// Bytes this handle addresses.
    ///
    /// A view reports its slot's extent, not the arena's. `fill_buffer` and
    /// `copy_from_buffer` both take `buffer.length()` as a range, so reporting
    /// the parent's length would let a blit over one slot run across every slot
    /// after it -- silent corruption of the exact kind §9.3 says rests on our
    /// offset arithmetic.
    pub fn length(&self) -> usize {
        match self.length {
            Some(len) => len,
            None => self.as_ref().length(),
        }
    }

    pub fn did_modify_range(&self, range: NSRange) {
        self.as_ref().didModifyRange(range);
    }

    pub fn set_label(&self, label: &str) {
        self.raw.setLabel(Some(&NSString::from_str(label)))
    }
}

impl AsRef<ProtocolObject<dyn MTLBuffer>> for Buffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }
}

impl<'a> From<&'a Buffer> for &'a MetalResource {
    fn from(val: &'a Buffer) -> Self {
        ProtocolObject::from_ref(val.as_ref())
    }
}

pub type BufferMap = HashMap<usize, Vec<Arc<Buffer>>>;
