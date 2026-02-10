/// Any collate gathers samples from one batch together.
pub trait Collate<T> {
    /// The type of the collate function's output.
    type Output;
    /// Take a batch of samples and collate them.
    fn collate(&self, batch: Vec<T>) -> Self::Output;
}

// Allow user to specify closure as collate function.
impl<T, F, O> Collate<T> for F
where
    F: Fn(Vec<T>) -> O,
{
    type Output = O;
    fn collate(&self, batch: Vec<T>) -> Self::Output {
        (self)(batch)
    }
}
