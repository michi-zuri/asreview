Start a review
==============

To start reviewing a dataset with ASReview LAB, you create a project containing
a dataset with records to screen. The project will contain your dataset,
settings, labeling decisions, and machine learning models.

To start a review project, you need to:

1. :doc:`start`.
2. Go to the *Reviews* if you are not already there
   (http://localhost:5000/reviews)
3. Upload, select, or choose a dataset to screen.
4. Verify the dataset with the charts. Ensure that the dataset completeness
   is sufficient.

Add Dataset
-----------

The first step in creating a project is to select a dataset. You can upload a
dataset from your computer, select a dataset from Discovery, or use a dataset
from a URL or DOI. When uploading a dataset from your computer, URL, or DOI,
ensure that the dataset is in a supported format. See :doc:`data` for extensive
information about the supported formats and metadata.

.. tip::

    You will benefit most from what active learning has to offer with
    :ref:`lab/data:High-quality data`.

From File
~~~~~~~~~

Drag and drop your file or select your file.


From URL (or DOI)
~~~~~~~~~~~~~~~~~

Provide a URL or a DOI to a dataset. Many data repositories are supported via
`Datahugger <https://github.com/J535D165/datahugger>`__. If the DOI points to
multiple files, you can select the specific file you want to use (e.g.,
`10.17605/OSF.IO/WDZH5 <https://doi.org/10.17605/OSF.IO/WDZH5>`__).

Click on *Download* to download and add the dataset to the project.

From Discovery
~~~~~~~~~~~~~~

Under Discovery, you can select existing datasets from the `SYNERGY dataset
<https://github.com/asreview/synergy-dataset>`__ or installed dataset
extensions. The SYNERGY dataset is a collection of fully labeled datasets that
can be used, but not exclusively, to benchmark the performance of active
learning models.

More options
------------

Under the dataset card, you find *Show options*. Clicking on *show options* will
open extra options for the review.


.. figure:: ../../images/setup_more_options.png
   :alt: ASReview LAB warmup


Add Tags
~~~~~~~~

You can add tags to your records to review. Tags are useful for organizing your
records afterwards or for data extraction. You can add tags and tag groups to
your review by clicking on the *Add tags* button. You can add multiple tags to a
tag group, and you can add multiple tags to a dataset. In the current version,
you can't delete tags, so be careful with the tags you add.

Each tag group can be configured in the *Customize* tab:

- **Single select**: when enabled, only one tag in the group can be selected and
  the tags are shown as radio buttons in the *Reviewer* interface. When
  disabled, the tags are shown as checkboxes and any number of them can be
  selected.
- **Required per decision**: a selection in the group can be required for
  *relevant* decisions, for *not relevant* decisions, or for both, using two
  independent toggles. This gives four combinations: never required, required
  only when marking a record relevant, required only when marking it not
  relevant, or always required. The matching decision button stays disabled
  until the requirement is met. Required groups are marked with an asterisk
  (``*``), explained by the legend *"\* a selection must be made in this
  group"*.
- **Checklist (require all options)**: for a multi-select group that is required
  for at least one decision, you can additionally require that *every* option is
  selected, turning the group into a checklist. This toggle is only available
  when single-select is off and at least one of the required toggles is on.
- **Free text**: each individual tag can opt in to an additional free-text input
  field, letting reviewers attach a short note to that specific tag.

In a single-select group the selected radio button can always be deselected by
clicking it again. Deselecting a required group leaves it empty, so the matching
decision button stays disabled until a selection is made again.

Tags are presented to you in the *Reviewer* interface as checkboxes or radio
buttons depending on the group configuration. The selected tags (and any
free-text additions) can be found in the **Collection** and during the export of
the dataset, where each tag is exported as ``tag_<group>_<value>`` and, if
provided, ``tag_<group>_<value>_text``. The collection only displays the tags
that were actually selected; empty optional groups show a placeholder. If an
invalid combination is encountered (for example more than one option selected in
a single-select group, which can happen when a project is edited across
versions; a required selection missing for the record's decision; or a checklist
group that is not fully checked), the record is still shown but with a warning,
and the **Collection** filter offers an *invalid tags* option to find these
records.

.. note::

   The on-disk tag format is backwards compatible. Older ASReview versions can
   still read and display tags created with these options; they simply ignore
   the single-select, required and free-text metadata and the per-tag free-text
   note.

Add Lists
~~~~~~~~~

Next to tags, you can configure *lists* in the *Customize* tab. A list is a
named container that lets a reviewer add an arbitrary number of free-text items
to a record. Typical uses are extracting outcomes, populations, or any other
open-ended set of values that varies per record.

Each list has a **name** and a **required for relevant** toggle. When the toggle
is on, at least one item must be added to the list before the record can be
marked *relevant*; required lists are marked with an asterisk (``*``), explained
by the legend *"\* at least one item must be added for allowing relevant
decision"*.

You can add several lists, and each record can collect multiple items per list.
Lists are shown and edited in the *Reviewer* interface between the tags and the
note of a record. Each item is a short free-text label and may not contain a
comma (``,``) or a semicolon (``;``). Every item is assigned a stable ``uuid``
identifier when it is created. Instead of a separate *add item* button there is
always an empty input row to type into; empty rows keep their position while you
edit and are dropped automatically when the list is saved.

Items are ordered by the moment they were created (oldest first). When editing
an item you can use the down-arrow button to reset its timestamp to now, which
moves it to the end of the list. No drag-and-drop reordering is needed.

Because a record can have many items per list, the items are stored in a
dedicated ``lists`` table in the project database. ``item_id`` is the primary
key (so items can later be referenced by foreign keys), and there is a unique
constraint on ``(record_id, list_id, name)`` so the same item cannot be added
twice to one list on a record. The columns are ``record_id``,
``list_id``, ``item_id``, ``name`` and ``created``. Both ``list_id`` and
``item_id`` are ``uuid4`` strings. The mapping from each ``list_id`` to its
display name (and required flag) is stored in a ``lists.json`` file in the
project folder, analogous to ``tags.json``.

Change AI Model
~~~~~~~~~~~~~~~

By default, ASReview LAB uses the ELAS ultra model. This is a fast and efficient
model that is trained on the SYNERGY dataset. You can change the model to a
different model by clicking on the dropdown button. You can select from the
following models:

- ELAS ultra
- ELAS multilingual
- ELAS heavy
- Custom

Most users will benefit from the ELAS ultra model and don't need to change the
model. The ELAS multilingual model is useful for datasets that are multilingual
or contain non-English records.

For more information about the models and the required :doc:`dory` extension,
see the :ref:`lab/models` page.


Prior Knowledge
~~~~~~~~~~~~~~~

Prior knowledge refers to records in your dataset that you already know are
relevant or irrelevant. Providing prior knowledge helps train the model during
the initial and subsequent iterations of the active learning cycle. The model
uses this information to generate an initial ranking of records in your dataset.

.. note::

  If your dataset includes :ref:`lab/data_labeled:Partially labeled data`,
  ASReview LAB will automatically use the labeled records as prior knowledge.

To add prior knowledge:

1. Click on *Search* to search your dataset by authors, keywords, titles, or a
   combination of these.
2. Enter your search terms and press *Enter*. Only the first 10 results will be
   displayed, so ensure your search terms are precise.
3. Review the record you were searching for and select the relevant or
   irrelevant label. You can also add tags to the record. Avoid labeling all
   items; select only those you intend to use as training data.
4. Close the search window or click on *Return* to return to the previous
   screen.

Providing accurate prior knowledge improves the model's performance and can
accelerate the review process.


Screen
------

Once you have selected a dataset and optionally added tags, changed the model,
or searched for prior knowledge, you can click on *Screen* to start the review. For
more tips on how to screen records, see :doc:`screening`.
