# Complementary component control

Added after the first symbolic-candidate model completed (shared1/10402,
perfect familiar and held accuracy), before any symbolic-candidate result
from the failing shared2 setting. Add a complementary `fixed_query` control
on the same four settings: encode the input query in the fixed source,
target, event slots and train the original scene encoder. Initialization,
data schedule, training duration, and evaluation remain the same.

If each separately learned encoder succeeds with a fixed counterpart while
the joint system fails, the joint learning/representation geometry becomes
a better target. This does not establish a unique mechanism: changing the
fixed counterpart changes the learning bias and objective geometry.
